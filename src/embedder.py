import json
import asyncio
import hashlib
import boto3
import tiktoken
from botocore.config import Config
from botocore.exceptions import ClientError
from cachetools import LRUCache
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langfuse import Langfuse
from src.config import settings

MAX_TOKENS = 8000
BATCH_SIZE = 10

# Connection pool - reuses HTTP connections across thread-pool calls
_client = boto3.client(
    "bedrock-runtime",
    region_name=settings.bedrock.region,
    config=Config(max_pool_connections=20),
)

# Tiktoken for input truncation - Titan v2 hard limit is 8192 tokens
_tokenizer = tiktoken.get_encoding("cl100k_base")

# LRU embedding cache - keyed on SHA-256 of the text
_cache = LRUCache(maxsize=1000)

_langfuse = Langfuse(
    public_key=settings.langfuse.public_key,
    secret_key=settings.langfuse.secret_key,
    host=settings.langfuse.host,
)

def _truncate(text: str) -> tuple[str, int]:
    tokens = _tokenizer.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    return _tokenizer.decode(tokens), len(tokens)

# Retry on throttling — waits 2s, 4s, 8s before giving up
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ClientError),
)
def _embed_one(text: str) -> tuple[list[float], int]:
    text, token_count = _truncate(text)
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = _client.invoke_model(
        modelId=settings.bedrock.embedding_model,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"], token_count

async def embed_texts(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    results = []
    total_tokens = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        
        # Split batch into cache hits and misses
        to_embed: list[tuple[int, str, str]] = []
        batch_results: list[list[float] | None] = [None] * len(batch)

        for j, text in enumerate(batch):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in _cache:
                batch_results[j] = _cache[key]
            else:
                to_embed.append((j, text, key))
        
        # Embed cache misses concurrently
        if to_embed:
            vectors_and_counts = await asyncio.gather(
                *[loop.run_in_executor(None, _embed_one, text) for _, text, _ in to_embed]
            )
            for (j, _, key), (vector, token_count) in zip(to_embed, vectors_and_counts):
                _cache[key] = vector
                batch_results[j] = vector
                total_tokens += token_count

        results.extend(batch_results)

    generation = _langfuse.generation(
        name=f"bedrock-embed",
        model=settings.bedrock.embedding_model,
        usage={"input": total_tokens, "unit": "TOKENS"},
    )
    generation.end()

    return results