from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # In this override impl we just need to add extra actions, we need to propagate attachment to the Choice since
        # in DeploymentTool they were propagated to the stage only as files. The main goal here is show pictures in chat
        # (DIAL Chat support special markdown to load pictures from DIAL bucket directly to the chat)
        # ---
        # 1. Call parent function `_execute` and get result
        result = await super()._execute(tool_call_params)
        
        # Parent always returns Message, so we can safely assume it's a Message
        assert isinstance(result, Message), "Expected Message from parent _execute"
        
        # 2. If attachments are present then filter only "image/png" and "image/jpeg"
        image_attachments = []
        if result.custom_content and result.custom_content.attachments:
            image_attachments = [
                att for att in result.custom_content.attachments
                if att.type in ["image/png", "image/jpeg"]
            ]
        
        # 3. Append then as content to choice in such format `f"\n\r![image]({attachment.url})\n\r")`
        image_markdown = ""
        for attachment in image_attachments:
            if attachment.url:
                image_markdown += f"\n\r![image]({attachment.url})\n\r"
                # Also add to choice so it appears in chat
                tool_call_params.choice.add_attachment(attachment)
        
        # 4. After iteration through attachment if message content is absent add such instruction:
        #    'The image has been successfully generated according to request and shown to user!'
        #    Sometimes models are trying to add generated pictures as well to content (choice), with this instruction
        #    we are notifing LLLM that it was done (but anyway sometimes it will try to add file 😅)
        current_content = str(result.content) if result.content else ""
        if image_markdown:
            if not current_content:
                current_content = "The image has been successfully generated according to request and shown to user!"
            current_content += image_markdown
        
        # Update the message content
        result.content = StrictStr(current_content) if current_content else None
        
        return result

    @property
    def deployment_name(self) -> str:
        # provide deployment name for model that you have added to DIAL Core config (dall-e-3)
        return "dall-e-3"

    @property
    def name(self) -> str:
        # provide self-descriptive name
        return "generate_image"

    @property
    def description(self) -> str:
        # provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return ("Generates images using DALL-E-3 based on a detailed text description. "
                "Use this tool when the user requests image generation, picture creation, or visual content. "
                "Provide a comprehensive and detailed prompt describing the desired image including subject, style, "
                "colors, composition, and any specific details. The generated image will be automatically displayed "
                "in the chat. Optional parameters: size (1024x1024, 1792x1024, or 1024x1792), quality (standard or hd), "
                "and style (vivid or natural).")

    @property
    def parameters(self) -> dict[str, Any]:
        # provide tool parameters JSON Schema:
        #  - prompt is string, description: "Extensive description of the image that should be generated.", required
        #  - there are 3 optional parameters: https://platform.openai.com/docs/guides/image-generation?image-generation-model=dall-e-3#customize-image-output
        #  - Sample: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/dall-e?tabs=dalle-3#call-the-image-generation-api
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated. Include details about subject, style, colors, composition, mood, and any specific elements."
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "description": "The size of the generated images. Must be one of 1024x1024, 1792x1024, or 1024x1792 pixels.",
                    "default": "1024x1024"
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "hd"],
                    "description": "The quality of the image that will be generated. hd creates images with finer details and greater consistency across the image.",
                    "default": "standard"
                },
                "style": {
                    "type": "string",
                    "enum": ["vivid", "natural"],
                    "description": "The style of the generated images. vivid produces hyper-real and dramatic images, natural produces more natural, less hyper-real looking images.",
                    "default": "vivid"
                }
            },
            "required": ["prompt"]
        }

