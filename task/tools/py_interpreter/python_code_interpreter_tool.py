import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        # 1. Set dial_endpoint
        self.dial_endpoint = dial_endpoint
        
        # 2. Set mcp_client
        self.mcp_client = mcp_client
        
        # 3. Set _code_execute_tool: Optional[MCPToolModel] as None at start, then iterate through `mcp_tool_models` and
        #    if any of tool model has the same same as `tool_name` then set _code_execute_tool as tool model
        self._code_execute_tool: Optional[MCPToolModel] = None
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break
        
        # 4. If `_code_execute_tool` is null then raise error (We cannot set up PythonCodeInterpreterTool without tool that executes code)
        if self._code_execute_tool is None:
            raise ValueError(f"Cannot set up PythonCodeInterpreterTool without tool '{tool_name}'. Available tools: {[t.name for t in mcp_tool_models]}")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        # 1. Create MCPClient
        mcp_client = await MCPClient.create(mcp_url)
        
        # 2. Get tools
        mcp_tool_models = await mcp_client.get_tools()
        
        # 3. Create PythonCodeInterpreterTool instance and return it
        return cls(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tool_models,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint
        )

    @property
    def show_in_stage(self) -> bool:
        # set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # provide `_code_execute_tool` name
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        # provide `_code_execute_tool` description
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        # provide `_code_execute_tool` parameters
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get `code` from arguments
        code = arguments.get("code")
        
        # 3. Get `session_id` from arguments (it is optional parameter, use get method)
        session_id = arguments.get("session_id")
        
        # 4. Get stage from `tool_call_params`
        stage = tool_call_params.stage
        
        # 5. Append content to stage: "## Request arguments: \n"
        stage.append_content("## Request arguments: \n")
        
        # 6. Append content to stage: `"```python\n\r{code}\n\r```\n\r"` it will show code in stage as python markdown
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        
        # 7. Append session to stage:
        #       - if `session_id` is present and not 0 then append to stage `f"**session_id**: {session_id}\n\r"`
        #       - otherwise append "New session will be created\n\r"
        if session_id and session_id != 0:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")
        
        # 8. Make tool call
        tool_result = await self.mcp_client.call_tool(self._code_execute_tool.name, arguments)
        
        # 9. Load retrieved response as json (️⚠️ here can be potential issues if you didn't properly implemented
        #    MCPClient tool call, it must return string)
        if isinstance(tool_result, str):
            result_data = json.loads(tool_result)
        else:
            # If it's already a dict, use it directly
            result_data = tool_result if isinstance(tool_result, dict) else {"content": str(tool_result)}
        
        # 10. Validate result with _ExecutionResult (it is full copy of https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/models.py)
        execution_result = _ExecutionResult(**result_data)
        
        # 11. If execution_result contains files we need to pool files from PyInterpreter and upload them to DIAL bucked:
        if execution_result.files:
            #       - Create Dial client
            client = Dial(base_url=self.dial_endpoint, api_key=tool_call_params.api_key)
            
            # Try to extract bucket ID from conversation_id or use it directly
            # DIAL file URLs follow pattern: files/{bucket_id}/uploads/{date}/{filename}
            # We'll try using conversation_id as bucket, or extract from a pattern
            bucket_id = None
            conversation_id = tool_call_params.conversation_id
            
            # Try to use conversation_id as bucket ID (it might be in the right format)
            if conversation_id and len(conversation_id) > 20:  # Bucket IDs seem to be long strings
                bucket_id = conversation_id
                print(f"[DEBUG] Using conversation_id as bucket_id: {bucket_id[:20]}...")
            else:
                # Generate a deterministic bucket ID from conversation_id if available
                import hashlib
                if conversation_id:
                    # Create a hash-based bucket ID from conversation_id
                    bucket_hash = hashlib.sha256(conversation_id.encode()).hexdigest()[:32]
                    bucket_id = bucket_hash
                    print(f"[DEBUG] Generated bucket_id from conversation_id: {bucket_id}")
            
            # Note: Files must be retrieved immediately while the session is still active
            # Sessions can expire quickly, so we process files right after execution
            print(f"[DEBUG] Processing {len(execution_result.files)} file(s) from execution result")
            
            #       - Iterated through files and:
            for file_ref in execution_result.files:
                try:
                    #           - get file name and mime_type and assign to appropriate variables
                    file_name = file_ref.name
                    mime_type = file_ref.mime_type
                    
                    #           - get resource with mcp client by URL from file (https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L429)
                    #           - Handle session expiration errors gracefully
                    #           - Files must be retrieved immediately while session is active
                    print(f"[DEBUG] Retrieving file resource: {file_ref.uri}")
                    try:
                        resource_content = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))
                        print(f"[DEBUG] Successfully retrieved file {file_name} ({len(resource_content) if isinstance(resource_content, (str, bytes)) else 'unknown'} bytes)")
                    except McpError as mcp_err:
                        # Check if it's a session expiration error
                        error_msg_str = str(mcp_err)
                        if "not found or has expired" in error_msg_str or ("Session" in error_msg_str and "expired" in error_msg_str):
                            error_msg = f"⚠️ Session expired before file '{file_name}' could be retrieved. This can happen if there's a delay between code execution and file retrieval. The file was generated but is no longer accessible. Please re-run the code to regenerate the file."
                            print(f"[WARNING] {error_msg}")
                            stage.append_content(f"**Warning**: {error_msg}\n\r")
                            # Continue to next file - don't fail completely
                            continue
                        else:
                            # Re-raise other MCP errors
                            error_msg = f"Error retrieving file {file_name}: {str(mcp_err)}"
                            print(f"[ERROR] {error_msg}")
                            stage.append_content(f"**Error**: {error_msg}\n\r")
                            raise
                    
                    #           - according to MCP binary resources must be encoded with base64 https://modelcontextprotocol.io/specification/2025-06-18/server/resources#binary-content
                    #             The get_resource method already handles base64 decoding for BlobResourceContents, so:
                    #             - For text files: if it's a string, encode to bytes; if already bytes, use as-is
                    #             - For binary files: get_resource already returns decoded bytes, so use as-is
                    if isinstance(resource_content, str):
                        # Text content - encode to bytes
                        file_bytes = resource_content.encode('utf-8')
                    elif isinstance(resource_content, bytes):
                        # Binary content - already decoded bytes from get_resource
                        file_bytes = resource_content
                    else:
                        # Fallback: convert to bytes
                        file_bytes = bytes(resource_content) if resource_content else b""
                    
                    # Ensure we have valid bytes
                    if not file_bytes:
                        error_msg = f"Failed to retrieve file content for {file_name}"
                        print(f"[ERROR] {error_msg}")
                        stage.append_content(f"**Warning**: {error_msg}\n\r")
                        continue
                    
                    #           - Prepare file for upload
                    # Construct upload URL using bucket_id if available
                    from datetime import datetime
                    
                    uploaded_file = None
                    upload_errors = []
                    
                    # Create file-like object
                    file_obj = io.BytesIO(file_bytes)
                    if hasattr(file_obj, 'seek'):
                        file_obj.seek(0)
                    
                    # Try with bucket_id if we have it
                    if bucket_id:
                        date_str = datetime.now().strftime('%Y-%m')
                        upload_url = f"files/{bucket_id}/uploads/{date_str}/{file_name}"
                        print(f"[DEBUG] Uploading file {file_name} ({len(file_bytes)} bytes, {mime_type}) to URL: {upload_url}")
                        
                        # Try tuple format with bucket-based URL
                        try:
                            uploaded_file = client.files.upload(upload_url, (file_name, file_bytes, mime_type))
                            print(f"[DEBUG] Upload successful using tuple (filename, bytes, mime_type) with bucket URL")
                        except Exception as e1:
                            upload_errors.append(f"tuple with bucket URL: {str(e1)}")
                            print(f"[DEBUG] Upload failed with bucket URL: {str(e1)}")
                            
                            # Try without mime_type
                            try:
                                uploaded_file = client.files.upload(upload_url, (file_name, file_bytes))
                                print(f"[DEBUG] Upload successful using tuple (filename, bytes) with bucket URL")
                            except Exception as e2:
                                upload_errors.append(f"tuple without mime_type: {str(e2)}")
                                print(f"[DEBUG] Upload failed: {str(e2)}")
                                
                                # Try just bytes
                                try:
                                    uploaded_file = client.files.upload(upload_url, file_bytes)
                                    print(f"[DEBUG] Upload successful using bytes with bucket URL")
                                except Exception as e3:
                                    upload_errors.append(f"bytes: {str(e3)}")
                                    print(f"[DEBUG] Upload failed: {str(e3)}")
                    
                    # If bucket-based upload failed, try simpler approaches
                    if uploaded_file is None:
                        print(f"[DEBUG] Trying alternative upload methods without bucket...")
                        
                        # Try with just filename
                        try:
                            uploaded_file = client.files.upload(file_name, (file_name, file_bytes, mime_type))
                            print(f"[DEBUG] Upload successful using tuple with filename as URL")
                        except Exception as e:
                            upload_errors.append(f"filename URL: {str(e)}")
                            print(f"[DEBUG] Upload failed: {str(e)}")
                    
                    # Since we can't upload to DIAL storage without a valid bucket ID,
                    # we'll use base64 data URIs to embed the image directly in the attachment
                    # This allows the image to be displayed without needing DIAL storage
                    
                    if uploaded_file is None:
                        # Upload failed - use base64 data URI instead
                        print(f"[INFO] Upload to DIAL storage failed, using base64 data URI for {file_name}")
                        
                        # Encode file bytes as base64
                        import base64
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        
                        # Create data URI: data:image/png;base64,{base64_data}
                        data_uri = f"data:{mime_type};base64,{base64_data}"
                        
                        # Create attachment with data URI
                        attachment = Attachment(
                            url=data_uri,
                            type=mime_type,
                            title=file_name
                        )
                        
                        # Add attachment to stage and choice
                        stage.add_attachment(attachment)
                        tool_call_params.choice.add_attachment(attachment)
                        print(f"[SUCCESS] Successfully attached file {file_name} using base64 data URI ({len(base64_data)} chars)")
                        continue
                    
                    # Get the URL from uploaded_file - it might be in different attributes
                    file_url = None
                    print(f"[DEBUG] Extracting URL from uploaded_file object (type: {type(uploaded_file)})")
                    if hasattr(uploaded_file, 'url'):
                        file_url = uploaded_file.url
                        print(f"[DEBUG] Found URL in .url attribute: {file_url}")
                    elif hasattr(uploaded_file, 'file_url'):
                        file_url = uploaded_file.file_url
                        print(f"[DEBUG] Found URL in .file_url attribute: {file_url}")
                    elif isinstance(uploaded_file, str):
                        file_url = uploaded_file
                        print(f"[DEBUG] Uploaded file is a string (URL): {file_url}")
                    elif isinstance(uploaded_file, dict):
                        file_url = uploaded_file.get('url') or uploaded_file.get('file_url')
                        print(f"[DEBUG] Found URL in dict: {file_url}")
                    else:
                        # Try to inspect the object
                        print(f"[DEBUG] Uploaded file object attributes: {dir(uploaded_file)}")
                        if hasattr(uploaded_file, '__dict__'):
                            print(f"[DEBUG] Uploaded file object __dict__: {uploaded_file.__dict__}")
                    
                    if not file_url:
                        # Fallback to base64 data URI if URL extraction fails
                        print(f"[WARNING] Could not extract URL from uploaded_file, falling back to base64 data URI")
                        import base64
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        file_url = f"data:{mime_type};base64,{base64_data}"
                        print(f"[DEBUG] Using base64 data URI as fallback")
                    
                    print(f"[DEBUG] Successfully extracted file URL: {file_url[:100]}..." if len(str(file_url)) > 100 else f"[DEBUG] Successfully extracted file URL: {file_url}")
                    
                    #           - Prepare Attachment with url, type (mime_type), and title (file_name)
                    attachment = Attachment(
                        url=file_url,
                        type=mime_type,
                        title=file_name
                    )
                    
                    #           - Add attachment to stage and also add this attachment to choice (it will be chown in both stage and choice)
                    stage.add_attachment(attachment)
                    tool_call_params.choice.add_attachment(attachment)
                    print(f"[SUCCESS] Successfully uploaded and attached file: {file_name}")
                    
                except Exception as e:
                    error_msg = f"Error processing file {file_ref.name}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    stage.append_content(f"**Error processing file {file_ref.name}**: {str(e)}\n\r")
                    # Continue with next file
                    continue
        
        # 12. Check if execution_result output present and if yes iterate through all output results and cut it length
        #     to 1000 chars, it is needed to avoid high costs and context window overload
        if execution_result.output:
            execution_result.output = [
                output[:1000] + "..." if len(output) > 1000 else output
                for output in execution_result.output
            ]
        
        # 13. Append to stage response f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r"
        stage.append_content(f"```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        
        # 14. Return execution result as string (model_dump_json method)
        return execution_result.model_dump_json()
