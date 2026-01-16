import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

# System prompt for Generation step
_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided document context.
Use only the information from the provided context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        # 1. Set endpoint
        self.endpoint = endpoint
        
        # 2. Set deployment_name
        self.deployment_name = deployment_name
        
        # 3. Set document_cache. DocumentCache is implemented, relate to it as to centralized Dict with file_url (as key),
        #    and indexed embeddings (as value), that have some autoclean. This cache will allow us to speed up RAG search.
        self.document_cache = document_cache
        
        # 4. Create SentenceTransformer and set is as `model` with:
        #   - model_name_or_path='all-MiniLM-L6-v2', it is self hosted lightwait embedding model.
        #     More info: https://medium.com/@rahultiwari065/unlocking-the-power-of-sentence-embeddings-with-all-minilm-l6-v2-7d6589a5f0aa
        #   - Optional! You can set it use CPU forcefully with `device='cpu'`, in case if not set up then will use GPU if it has CUDA cores
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 5. Create RecursiveCharacterTextSplitter as `text_splitter` with:
        #   - chunk_size=500
        #   - chunk_overlap=50
        #   - length_function=len
        #   - separators=["\n\n", "\n", ". ", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        # set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # provide self-descriptive name
        return "rag_search"

    @property
    def description(self) -> str:
        # provide tool description that will help LLM to understand when to use this tools and cover 'tricky' moments
        return ("Performs semantic search on uploaded documents to find relevant information and answer questions. "
                "Use this tool when you need to search through large documents (especially when file_content_extraction "
                "shows pagination), or when you need to find specific information in a document. This tool is more efficient "
                "than extracting full file content for large documents. It uses semantic search to find the most relevant "
                "chunks of text related to your query. Supports PDF, TXT, CSV, and HTML files.")

    @property
    def parameters(self) -> dict[str, Any]:
        # provide tool parameters JSON Schema:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "URL of the file to search in"
                }
            },
            "required": ["request", "file_url"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get `request` from arguments
        request = arguments.get("request")
        
        # 3. Get `file_url` from arguments
        file_url = arguments.get("file_url")
        
        # 4. Get stage from `tool_call_params`
        stage = tool_call_params.stage
        
        # 5. Append content to stage: "## Request arguments: \n"
        stage.append_content("## Request arguments: \n")
        
        # 6. Append content to stage: `f"**Request**: {request}\n\r"`
        stage.append_content(f"**Request**: {request}\n\r")
        
        # 7. Append content to stage: `f"**File URL**: {file_url}\n\r"`
        stage.append_content(f"**File URL**: {file_url}\n\r")
        
        # 8. Create `cache_document_key`, it is string from `conversation_id` and `file_url`, with such key we guarantee
        #    access to cached indexes for one particular conversation,
        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        
        # 9. Get from `document_cache` by `cache_document_key` a cache
        cached_data = self.document_cache.get(cache_document_key)
        
        # 10. If cache is present then set it as `index, chunks = cached_data` (cached_data is retrieved cache from 9 step),
        #     otherwise:
        if cached_data:
            index, chunks = cached_data
        else:
            #       - Create DialFileContentExtractor and extract text by `file_url` as `text_content`
            extractor = DialFileContentExtractor(endpoint=self.endpoint, api_key=tool_call_params.api_key)
            text_content = extractor.extract_text(file_url)
            
            #       - If no `text_content` then appen to stage info about it ans return the string with the error that file content is not found
            if not text_content:
                stage.append_content("Error: File content not found.\n\r")
                return "Error: File content not found."
            
            #       - Create `chunks` with `text_splitter`
            chunks = self.text_splitter.split_text(text_content)
            
            #       - Create `embeddings` with `model`
            embeddings = self.model.encode(chunks, convert_to_numpy=True)
            
            #       - Create IndexFlatL2 with `384` dimensions as `index` (more about IndexFlatL2 https://shayan-fazeli.medium.com/faiss-a-quick-tutorial-to-efficient-similarity-search-595850e08473)
            index = faiss.IndexFlatL2(384)
            
            #       - Add to `index` np.array with created embeddings as type 'float32'
            index.add(np.array(embeddings, dtype='float32'))
            
            #       - Add to `document_cache`
            self.document_cache.set(cache_document_key, index, chunks)
        
        # 11. Prepare `query_embedding` with model. You need to encode request as type 'float32'
        query_embedding = self.model.encode([request], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # 12. Through created index make search with `query_embedding`, `k` set as 3. As response we expect tuple of
        #     `distances` and `indices`
        distances, indices = index.search(query_embedding, k=3)
        
        # 13. Now you need to iterate through `indices[0]` and and by each idx get element from `chunks`, result save as `retrieved_chunks`
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        
        # 14. Make augmentation
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        
        # 15. Append content to stage: "## RAG Request: \n"
        stage.append_content("## RAG Request: \n")
        
        # 16. Append content to stage: `ff"```text\n\r{augmented_prompt}\n\r```\n\r"` (will be shown as markdown text)
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        
        # 17. Append content to stage: "## Response: \n"
        stage.append_content("## Response: \n")
        
        # 18. Now make Generation with AsyncDial (don't forget about api_version '2025-01-01-preview', provide LLM with system prompt and augmented prompt and:
        #   - stream response to stage (user in real time will be able to see what the LLM responding while Generation step)
        #   - collect all content (we need to return it as tool execution result)
        api_key = tool_call_params.api_key
        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')
        
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt}
        ]
        
        content = ""
        # AsyncDial requires 'deployment_name' as keyword argument
        # The create() method returns a coroutine that needs to be awaited to get the stream
        stream = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            stream=True
        )
        
        async for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta if chunk.choices[0].delta else None
                if delta and hasattr(delta, 'content') and delta.content:
                    chunk_content = delta.content
                    stage.append_content(chunk_content)
                    content += chunk_content
        
        # 19. return collected content
        return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        # make prompt augmentation
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
        augmented_prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {request}

Answer based only on the provided context. If the context doesn't contain enough information, say so."""
        return augmented_prompt
