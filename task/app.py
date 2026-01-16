import os
import sys
from pathlib import Path

# Add parent directory to path so we can import task module when running directly
# This needs to happen before any task.* imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
# from task.tools.deployment.web_search_tool import WebSearchTool  # Disabled - doesn't work with this endpoint
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-sonnet-3-7')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        # 1. Create list of BaseTool
        tools: list[BaseTool] = []
        
        # 2. Create MCPClient
        mcp_client = await MCPClient.create(url)
        
        # 3. Get tools, iterate through them and add them to created list as MCPTool where the client will be created
        #    MCPClient and mcp_tool_model will be the tool itself (see what `mcp_client.get_tools` returns).
        mcp_tool_models = await mcp_client.get_tools()
        for mcp_tool_model in mcp_tool_models:
            tools.append(MCPTool(client=mcp_client, mcp_tool_model=mcp_tool_model))
        
        # 4. Return created tool list
        return tools

    async def _create_tools(self) -> list[BaseTool]:
        # 1. Create list of BaseTool
        # ---
        # At the beginning this list can be empty. We will add here tools after they will be implemented
        # ---
        tools: list[BaseTool] = []
        
        # Step 2: Add FileContentExtractionTool with DIAL_ENDPOINT
        tools.append(FileContentExtractionTool(endpoint=DIAL_ENDPOINT))
        
        # Step 3b: Image Generation Tool
        tools.append(ImageGenerationTool(endpoint=DIAL_ENDPOINT))
        
        # Step 3b (part 5): Web Search Tool (deployment tool)
        # NOTE: This tool doesn't work with GPT-4o through this endpoint - it triggers content filter errors
        # because the endpoint doesn't support web_search tool type. Use DuckDuckGo MCP server (Step 4) instead.
        # tools.append(WebSearchTool(endpoint=DIAL_ENDPOINT))
        
        # Step 3a: RAG Tool
        document_cache = DocumentCache.create()
        tools.append(RagTool(endpoint=DIAL_ENDPOINT, deployment_name=DEPLOYMENT_NAME, document_cache=document_cache))
        
        # Step 4: MCP Tools (DuckDuckGo web search)
        try:
            mcp_tools = await self._get_mcp_tools("http://localhost:8051/mcp")
            tools.extend(mcp_tools)
        except Exception as e:
            print(f"Warning: Failed to get MCP tools from http://localhost:8051/mcp: {e}. MCP tools will not be available.")
        
        # Step 5: Python Code Interpreter Tool
        try:
            python_interpreter_tool = await PythonCodeInterpreterTool.create(
                mcp_url="http://localhost:8050/mcp",
                tool_name="execute_code",
                dial_endpoint=DIAL_ENDPOINT
            )
            tools.append(python_interpreter_tool)
        except Exception as e:
            print(f"Warning: Failed to create PythonCodeInterpreterTool: {e}. The tool will not be available.")
        
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        print(f"[DEBUG] chat_completion called, DIAL_ENDPOINT={DIAL_ENDPOINT}, DEPLOYMENT_NAME={DEPLOYMENT_NAME}")
        # 1. If `self.tools` are absent then call `_create_tools` method and assign to the `self.tools`
        if not self.tools:
            print("[DEBUG] Creating tools...")
            self.tools = await self._create_tools()
            print(f"[DEBUG] Created {len(self.tools)} tools")
        
        # 2. Create `choice` (`with response.create_single_choice() as choice:`) and:
        print("[DEBUG] Creating choice and agent...")
        with response.create_single_choice() as choice:
            #   - Create GeneralPurposeAgent with:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools
            )
            print("[DEBUG] Calling handle_request...")
            #   - call `handle_request` on created agent with:
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response
            )
            print("[DEBUG] handle_request completed")

# 1. Create DIALApp
app = DIALApp()

# 2. Create GeneralPurposeAgentApplication
agent_app = GeneralPurposeAgentApplication()

# 3. Add to created DIALApp chat_completion with:
#       - deployment_name="general-purpose-agent"
#       - impl=agent_app
# Use add_chat_completion method: first argument is deployment name, second is ChatCompletion instance
app.add_chat_completion("general-purpose-agent", agent_app)

# 4. Run it with uvicorn: `uvicorn.run({CREATED_DIAL_APP}, port=5030, host="0.0.0.0")`
if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")
