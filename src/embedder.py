import json
import asyncio
import hashlib
import boto3
import tiktoken
from abc import ABC, abstractmethod
from botocore.config import Config
from botocore.exceptions import ClientError
from cachetools import LRUCache
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langfuse import Langfuse
from src.config import settings

BATCH_SIZE = 10

_langfuse = Langfuse(
    public_key=settings.langfuse.public_key,
    secret_key=settings.langfuse.secret_key,
    host=settings.langfuse.host,
)

# LRU embedding cache — shared across providers, keyed on SHA-256 of text
_cache: LRUCache = LRUCache(maxsize=1000)


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text."""
        pass


class BedrockEmbedder(BaseEmbedder):
    _MAX_TOKENS = 8000

    def __init__(self) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.bedrock.region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(max_pool_connections=20),
        )
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _truncate(self, text: str) -> tuple[str, int]:
        tokens = self._tokenizer.encode(text)
        if len(tokens) > self._MAX_TOKENS:
            tokens = tokens[:self._MAX_TOKENS]
        return self._tokenizer.decode(tokens), len(tokens)

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ClientError),
    )
    def _embed_one(self, text: str) -> tuple[list[float], int]:
        text, token_count = self._truncate(text)
        body = json.dumps({
            "inputText": text,
            "dimensions": settings.embedding_dimensions,
            "normalize": True,
        })
        response = self._client.invoke_model(
            modelId=settings.bedrock.embedding_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"], token_count

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        results: list[list[float]] = []
        total_tokens = 0
        generation = _langfuse.generation(
            name="bedrock-embed",
            model=settings.bedrock.embedding_model,
        )
        try:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                to_embed: list[tuple[int, str, str]] = []
                batch_results: list[list[float] | None] = [None] * len(batch)

                for j, text in enumerate(batch):
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if key in _cache:
                        batch_results[j] = _cache[key]
                    else:
                        to_embed.append((j, text, key))

                for idx, (j, text, key) in enumerate(to_embed):
                    vector, token_count = await loop.run_in_executor(None, self._embed_one, text)
                    _cache[key] = vector
                    batch_results[j] = vector
                    total_tokens += token_count

                results.extend(batch_results)
        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            _langfuse.flush()
            raise

        generation.end(usage={"input": total_tokens, "unit": "TOKENS"})
        _langfuse.flush()
        return results


class OpenRouterEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter.api_key,
        )

    def _embed_one(self, text: str) -> tuple[list[float], int]:
        response = self._client.embeddings.create(
            model=settings.openrouter.embedding_model,
            input=text,
            encoding_format="float",
            dimensions=settings.embedding_dimensions,
        )
        token_count = response.usage.prompt_tokens if response.usage else 0
        return response.data[0].embedding, token_count

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        results: list[list[float]] = []
        total_tokens = 0
        generation = _langfuse.generation(
            name="openrouter-embed",
            model=settings.openrouter.embedding_model,
        )
        try:
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                to_embed: list[tuple[int, str, str]] = []
                batch_results: list[list[float] | None] = [None] * len(batch)

                for j, text in enumerate(batch):
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if key in _cache:
                        batch_results[j] = _cache[key]
                    else:
                        to_embed.append((j, text, key))

                for j, text, key in to_embed:
                    vector, token_count = await loop.run_in_executor(None, self._embed_one, text)
                    _cache[key] = vector
                    batch_results[j] = vector
                    total_tokens += token_count

                results.extend(batch_results)
        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            _langfuse.flush()
            raise

        generation.end(usage={"input": total_tokens, "unit": "TOKENS"})
        _langfuse.flush()
        return results


def get_embedder() -> BaseEmbedder:
    if settings.llm_provider == "openrouter":
        return OpenRouterEmbedder()
    return BedrockEmbedder()


_embedder: BaseEmbedder | None = None


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Module-level convenience — delegates to the provider selected in settings."""
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return await _embedder.embed_texts(texts)