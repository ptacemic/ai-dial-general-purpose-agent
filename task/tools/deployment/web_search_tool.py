from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class WebSearchTool(DeploymentTool):
    """
    Web search tool that uses a model's built-in web search capability.
    This tool performs web searches and returns relevant information from the internet.
    Based on the pattern from: https://github.com/khshanovskyi/ai-simple-agent/blob/completed/task/tools/web_search.py
    
    NOTE: This tool may not work with all models/endpoints. If you encounter content filter errors
    or unsupported tool errors, the model/endpoint likely doesn't support web_search tool type.
    In that case, use the DuckDuckGo MCP server (Step 4) as a free alternative for web search.
    """

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # Override to perform web search and return results as text
        # The agent will use these results to create a revised prompt for DALL-E-3
        import json
        from aidial_client import AsyncDial
        from aidial_sdk.chat_completion import Message, Role
        from pydantic import StrictStr
        
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get `prompt` from arguments (the search query)
        prompt = arguments.get("prompt", "")
        
        # 3. Create AsyncDial client
        api_key = tool_call_params.api_key
        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')
        
        # 4. Prepare messages - simple user message with search query
        # No system prompt to avoid content filter issues, just the search query
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 5. Call chat completions with web search enabled via tools
        # This will perform the search and return results as text
        stage = tool_call_params.stage
        content = ""
        
        try:
            # Use web_search tool type - this should perform the search and return results
            stream = await client.chat.completions.create(
                messages=messages,
                deployment_name=self.deployment_name,
                stream=True,
                tools=[
                    {
                        "type": "web_search"
                    }
                ]
            )
            
            # 6. Collect content (search results)
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta if chunk.choices[0].delta else None
                    if delta:
                        # Collect content (the search results)
                        if hasattr(delta, 'content') and delta.content:
                            chunk_content = delta.content
                            stage.append_content(chunk_content)
                            content += chunk_content
        except Exception as e:
            error_str = str(e)
            # Check if it's a content filter error (which indicates web_search tool type is not supported)
            if "content_filter" in error_str or "safety system" in error_str.lower() or "ResponsibleAIPolicyViolation" in error_str:
                error_msg = (
                    "Web search is not available through this deployment tool. "
                    "The endpoint does not support the web_search tool type for GPT-4o. "
                    "Please inform the user that web search functionality requires implementing "
                    "Step 4 (DuckDuckGo MCP server) from the README, which provides free web search capabilities. "
                    "For now, you can proceed without web search or ask the user to provide the information directly."
                )
            else:
                error_msg = f"Error performing web search: {error_str}"
            
            stage.append_content(error_msg)
            content = error_msg
        
        # 7. Return the search results as text content
        # The agent will use this to create a revised prompt for image generation
        message = Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            content=StrictStr(content) if content else None
        )
        
        return message

    @property
    def system_prompt(self) -> str | None:
        # No system prompt - just return raw search results
        # The agent will use these results to create prompts for other tools (like DALL-E-3)
        return None

    @property
    def deployment_name(self) -> str:
        # Use gpt-4o as it supports web search capabilities
        return "gpt-4o"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return ("Performs web search to find current information, news, facts, or any content from the internet. "
                "Use this tool when you need to search for recent information, current events, real-time data, "
                "or any information that requires accessing the web. The tool will search the web and return "
                "relevant results with sources. Use this for queries about current events, recent news, "
                "real-time information, or when you need to verify or find information that may have changed recently.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The search query or question to search for on the web. Be specific and clear about what information you need."
                }
            },
            "required": ["prompt"]
        }

    @property
    def tool_parameters(self) -> dict[str, Any]:
        # Return empty dict since we're handling tools directly in _execute
        return {}
