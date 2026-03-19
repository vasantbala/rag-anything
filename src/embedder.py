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
BATCH_SIZE = 3

# Connection pool - reuses HTTP connections across thread-pool calls
_client = boto3.client(
    "bedrock-runtime",
    region_name=settings.bedrock.region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
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

# Retry on throttling
@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60),
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
    obs = _langfuse.start_observation(
        name="bedrock-embed",
        as_type="embedding",
        model=settings.bedrock.embedding_model,
    )

    try:
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
            
            # Embed cache misses sequentially to avoid throttling
            if to_embed:
                vectors_and_counts = []
                for _, text, _ in to_embed:
                    result = await loop.run_in_executor(None, _embed_one, text)
                    vectors_and_counts.append(result)
                for (j, _, key), (vector, token_count) in zip(to_embed, vectors_and_counts):
                    _cache[key] = vector
                    batch_results[j] = vector
                    total_tokens += token_count

            results.extend(batch_results)
    except Exception as e:
        obs.update(level="ERROR", status_message=str(e))
        obs.end()
        _langfuse.flush()
        raise

    obs.update(usage_details={"input": total_tokens})
    obs.end()
    _langfuse.flush()

    return results