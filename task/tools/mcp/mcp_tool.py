import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class MCPTool(BaseTool):

    def __init__(self, client: MCPClient, mcp_tool_model: MCPToolModel):
        # 1. Set client
        self.client = client
        # 2. Set mcp_tool_model
        self.mcp_tool_model = mcp_tool_model

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get content with mcp client tool call
        stage = tool_call_params.stage
        
        # Append request arguments to stage (if show_in_stage is True, this is already done by base class)
        # But we can add additional formatting here if needed
        
        try:
            # Call the MCP tool
            content = await self.client.call_tool(self.mcp_tool_model.name, arguments)
            
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content)
            
        except Exception as e:
            error_msg = f"Error calling MCP tool {self.mcp_tool_model.name}: {str(e)}"
            stage.append_content(error_msg)
            return error_msg
        
        # 3. Append retrieved content to stage
        # Format it nicely like other tools do
        stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        
        # 4. return content
        return content

    @property
    def name(self) -> str:
        # provide name from mcp_tool_model
        return self.mcp_tool_model.name

    @property
    def description(self) -> str:
        # provide description from mcp_tool_model
        return self.mcp_tool_model.description

    @property
    def parameters(self) -> dict[str, Any]:
        # provide parameters from mcp_tool_model
        return self.mcp_tool_model.parameters
