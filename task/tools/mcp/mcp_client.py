import base64
from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create MCPClient (connection is lazy)"""
        # 1. Create instance of MCPClient with `cls`
        instance = cls(mcp_server_url)
        
        # 2. Don't connect immediately - connection will be lazy (when needed)
        # This allows the tool to be created even if the server isn't running yet
        
        # 3. return created instance
        return instance

    async def connect(self):
        """Connect to MCP server"""
        # 1. Check if session is present, if yes just return to finsh execution
        if self.session is not None:
            return
        
        try:
            # 2. Call `streamablehttp_client` method with `server_url` and set as `self._streams_context`
            self._streams_context = streamablehttp_client(self.server_url)
            
            # 3. Enter `self._streams_context`, result set as `read_stream, write_stream, _`
            read_stream, write_stream, _ = await self._streams_context.__aenter__()
            
            # 4. Create ClientSession with streams from above and set as `self._session_context`
            self._session_context = ClientSession(read_stream, write_stream)
            
            # 5. Enter `self._session_context` and set as self.session
            self.session = await self._session_context.__aenter__()
            
            # 6. Initialize session and print its result to console
            init_result = await self.session.initialize()
            print(f"MCP Client initialized: {init_result}")
        except Exception as e:
            # If connection fails, clean up and re-raise
            print(f"Failed to connect to MCP server at {self.server_url}: {e}")
            # Clean up any partial state
            self.session = None
            self._session_context = None
            self._streams_context = None
            raise


    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        # Get and return MCP tools as list of MCPToolModel
        if self.session is None:
            await self.connect()
        
        tools_result = await self.session.list_tools()
        tools = []
        
        for tool in tools_result.tools:
            # Convert MCP tool to MCPToolModel
            tool_model = MCPToolModel(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            )
            tools.append(tool_model)
        
        return tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        # Make tool call and return its result. Do it in proper way (it returns array of content and you need to handle it properly)
        if self.session is None:
            await self.connect()
        
        result: CallToolResult = await self.session.call_tool(tool_name, arguments=tool_args)
        
        # Handle the result - it contains an array of content items
        # We need to combine all text content into a single string
        # For Python code interpreter, this should be JSON, so we combine and return as string
        content_parts = []
        for content_item in result.content:
            if isinstance(content_item, TextContent):
                content_parts.append(content_item.text)
            elif hasattr(content_item, 'text'):
                content_parts.append(content_item.text)
            else:
                # If it's not text, convert to string
                content_parts.append(str(content_item))
        
        # Join all content parts and return as string
        # This should be JSON for Python code interpreter tool
        combined_content = ''.join(content_parts)
        return combined_content

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        # Get and return resource. Resources can be returned as TextResourceContents and BlobResourceContents, you
        #      need to return resource value (text or blob)
        if self.session is None:
            await self.connect()
        
        result: ReadResourceResult = await self.session.read_resource(uri=str(uri))
        
        # Handle the result - it contains contents which is a list
        if result.contents:
            content_item = result.contents[0]  # Get first content item
            
            if isinstance(content_item, TextResourceContents):
                return content_item.text
            elif isinstance(content_item, BlobResourceContents):
                # BlobResourceContents has base64 encoded data
                return base64.b64decode(content_item.blob)
            else:
                # Fallback: try to get text or blob attribute
                if hasattr(content_item, 'text'):
                    return content_item.text
                elif hasattr(content_item, 'blob'):
                    return base64.b64decode(content_item.blob)
        
        return b""  # Return empty bytes if no content

    async def close(self):
        """Close connection to MCP server"""
        # 1. Close `self._session_context`
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing session context: {e}")
        
        # 2. Close `self._streams_context`
        if self._streams_context is not None:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing streams context: {e}")
        
        # 3. Set session, _session_context and _streams_context as None
        self.session = None
        self._session_context = None
        self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False

