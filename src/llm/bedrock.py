import asyncio
import boto3

from src.llm.base import LLMProvider, LLMResponse
from src.config import settings


class BedrockProvider(LLMProvider):
    """LLM provider backed by Amazon Bedrock using the Converse API.

    boto3's converse() is synchronous; it is dispatched to a thread pool so the
    async event loop is never blocked.
    """

    def __init__(self) -> None:
        kwargs: dict = {"region_name": settings.bedrock.region}
        if settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self._client = boto3.client("bedrock-runtime", **kwargs)
        self.model_id = settings.bedrock.llm_model

    def _converse_sync(self, system: list[dict], conversation: list[dict]) -> dict:
        return self._client.converse(
            modelId=self.model_id,
            system=system,
            messages=conversation,
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
        )

    async def complete(self, messages: list[dict]) -> LLMResponse:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        conversation = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]

        # Run blocking boto3 call off the event loop
        response = await asyncio.to_thread(self._converse_sync, system, conversation)

        return LLMResponse(
            text=response["output"]["message"]["content"][0]["text"],
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
        )
