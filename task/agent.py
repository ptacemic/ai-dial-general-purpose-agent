import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        # Prepare tools_dict where key will be tool name and value tool itself
        self._tools_dict = {tool.name: tool for tool in tools}
        # Create dict with `state` name. Inside this dict we need to add `TOOL_CALL_HISTORY_KEY` with empty array
        self.state = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        print(f"[DEBUG] handle_request called, deployment={deployment_name}, endpoint={self.endpoint}")
        # Create AsyncDial, don't forget to provide endpoint as base_url and api_key
        api_key = getattr(request, 'api_key', None) or getattr(request, 'headers', {}).get('authorization', '').replace('Bearer ', '')
        api_version = getattr(request, 'api_version', '2025-01-01-preview')
        print(f"[DEBUG] Creating AsyncDial client, api_key present: {bool(api_key)}")
        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version=api_version)
        
        # Create `chunks` with created AsyncDial client
        messages = self._prepare_messages(request.messages)
        tool_schemas = [tool.schema for tool in self.tools]
        print(f"[DEBUG] Calling chat.completions.create with {len(messages)} messages, {len(tool_schemas)} tools")
        # AsyncDial requires 'deployment_name' as keyword argument
        # The create() method returns a coroutine that needs to be awaited to get the stream
        chunks = await client.chat.completions.create(
            messages=messages,
            tools=tool_schemas,
            deployment_name=deployment_name,
            stream=True
        )
        print("[DEBUG] Starting to stream chunks...")
        
        # Create tool_call_index_map and content
        tool_call_index_map: dict[int, dict[str, Any]] = {}
        content = ""
        
        # Make async loop through `chunks` and collect content, tool calls
        chunk_count = 0
        async for chunk in chunks:
            chunk_count += 1
            if chunk_count == 1:
                print(f"[DEBUG] Received first chunk")
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta if chunk.choices[0].delta else None
                if delta:
                    # If delta content is present then append this content to `choice`
                    if hasattr(delta, 'content') and delta.content:
                        choice.append_content(delta.content)
                        content += delta.content
                    
                    # If delta has tool_calls
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            index = tool_call_delta.index
                            
                            # If tool call has `id` (first chunk of tool call)
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                tool_call_index_map[index] = {
                                    'id': tool_call_delta.id,
                                    'type': 'function',
                                    'function': {
                                        'name': '',
                                        'arguments': ''
                                    }
                                }
                            
                            # Otherwise: get by tool call delta `index` from the `tool_call_index_map`
                            if index in tool_call_index_map:
                                tool_call = tool_call_index_map[index]
                                
                                # Check if provided tool_call_delta contains `function`
                                if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                    if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                        tool_call['function']['name'] = tool_call_delta.function.name
                                    
                                    # Get `arguments` (if not present set them as empty string)
                                    argument_chunk = getattr(tool_call_delta.function, 'arguments', '') or ''
                                    tool_call['function']['arguments'] += argument_chunk
        
        print(f"[DEBUG] Finished streaming, received {chunk_count} chunks, content length: {len(content)}, tool_calls: {len(tool_call_index_map) if tool_call_index_map else 0}")
        # Create `assistant_message`, with role, content and tool_calls
        tool_calls = None
        if tool_call_index_map:
            tool_calls = []
            for tool_call_data in tool_call_index_map.values():
                # Use validate method to create ToolCall
                tool_call = ToolCall.validate(tool_call_data)
                tool_calls.append(tool_call)
        
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content if content else None,
            tool_calls=tool_calls
        )
        
        # Check if `assistant_message` contains `tool_calls`
        if assistant_message.tool_calls:
            # Get conversation_id from request headers
            headers = getattr(request, 'headers', {})
            conversation_id = headers.get('x-conversation-id', '')
            
            # Create `tasks` list
            tasks = [
                self._process_tool_call(tool_call, choice, api_key, conversation_id)
                for tool_call in assistant_message.tool_calls
            ]
            
            # Gather tasks with asyncio
            tool_messages = await asyncio.gather(*tasks)
            
            # To the `state` to `TOOL_CALL_HISTORY_KEY` append `assistant_message` as dict
            self.state[TOOL_CALL_HISTORY_KEY].append(assistant_message.dict(exclude_none=True))
            
            # Extend the `state` `TOOL_CALL_HISTORY_KEY` with tool_messages
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)
            
            # Finally make recursive call
            return await self.handle_request(deployment_name, choice, request, response)
        
        # We don't have any tool calls and ready to finish user request
        # Set choice with `state`
        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        # Unpack messages with `unpack_messages` method
        unpacked_messages = unpack_messages(messages, self.state.get(TOOL_CALL_HISTORY_KEY, []))
        # Insert as first message the `system_prompt`
        unpacked_messages.insert(0, {"role": Role.SYSTEM.value, "content": self.system_prompt})
        # Print history: iterate through unpacked messages and print as json
        for msg in unpacked_messages:
            print(json.dumps(msg))
        # Return unpacked messages
        return unpacked_messages

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        # Get tool name from tool_call function name
        tool_name = tool_call.function.name
        # Open Stage with StageProcessor
        stage = StageProcessor.open_stage(choice, tool_name)
        # Get tool from `_tools_dict` by tool name
        tool = self._tools_dict.get(tool_name)
        if not tool:
            stage.append_content(f"Error: Tool '{tool_name}' not found")
            StageProcessor.close_stage_safely(stage)
            return {
                "role": Role.TOOL.value,
                "content": f"Error: Tool '{tool_name}' not found",
                "tool_call_id": tool_call.id
            }
        
        # If tool show_in_stage is true then append request arguments
        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r")
            stage.append_content("## Response: \n")
        
        # Execute tool
        tool_call_params = ToolCallParams(
            tool_call=tool_call,
            stage=stage,
            choice=choice,
            api_key=api_key,
            conversation_id=conversation_id
        )
        tool_message = await tool.execute(tool_call_params)
        
        # Close stage with StageProcessor
        StageProcessor.close_stage_safely(stage)
        
        # Return tool message as dict and don't forget to exclude none
        return tool_message.dict(exclude_none=True)
