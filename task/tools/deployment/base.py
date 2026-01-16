import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    @property
    def system_prompt(self) -> str | None:
        """Optional system prompt for deployment tools. Override in subclasses if needed."""
        return None

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get `prompt` from arguments (by default we provide `prompt` for each deployment tool, use this param name as standard)
        prompt = arguments.get("prompt", "")
        
        # 3. Delete `prompt` from `arguments` (there can be provided additional parameters and `prompt` will be added
        #    as user message content and other parameters as `custom_fields`)
        custom_fields = {k: v for k, v in arguments.items() if k != "prompt"}
        
        # 4. Create AsyncDial client (api_version is 2025-01-01-preview)
        api_key = tool_call_params.api_key
        client = AsyncDial(base_url=self.endpoint, api_key=api_key, api_version='2025-01-01-preview')
        
        # 5. Call chat completions with:
        #   - messages (here will be just user message. Optionally, in this class you can add system prompt `property`
        #     and if any deployment tool provides system prompt then we need to set it as first message (system prompt))
        #   - stream it
        #   - deployment_name
        #   - extra_body with `custom_fields` https://dialx.ai/dial_api#operation/sendChatCompletionRequest (last request param in documentation)
        #   - **self.tool_parameters (will load all tool parameters that were set up in deployment tools as params, like
        #     `top_p`, `temperature`, etc...)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        extra_body = {}
        if custom_fields:
            extra_body["custom_fields"] = custom_fields
        
        stream = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            stream=True,
            extra_body=extra_body if extra_body else None,
            **self.tool_parameters
        )
        
        # 6. Collect content and it to stage, also, collect custom_content -> attachments and if they are present add
        #    them to stage as attachment as well
        stage = tool_call_params.stage
        content = ""
        attachments = []
        
        async for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta if chunk.choices[0].delta else None
                if delta:
                    # Collect content
                    if hasattr(delta, 'content') and delta.content:
                        chunk_content = delta.content
                        stage.append_content(chunk_content)
                        content += chunk_content
                    
                    # Collect custom_content -> attachments
                    if hasattr(delta, 'custom_content') and delta.custom_content:
                        if hasattr(delta.custom_content, 'attachments') and delta.custom_content.attachments:
                            for attachment in delta.custom_content.attachments:
                                attachments.append(attachment)
                                stage.add_attachment(attachment)
        
        # 7. Return Message with tool role, content, custom_content and tool_call_id
        custom_content = None
        if attachments:
            custom_content = CustomContent(attachments=attachments)
        
        message = Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            content=StrictStr(content) if content else None,
            custom_content=custom_content
        )
        
        return message
