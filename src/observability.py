from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langfuse import Langfuse

from src.config import settings
from src.llm.base import LLMResponse
from src.llm.factory import get_llm_provider

if TYPE_CHECKING:
    from src.retriever import RetrievalResult


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    return Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
    )


async def traced_llm_call(
    question: str,
    messages: list[dict],
    retrieval_result: RetrievalResult,
    user_id: str,
) -> LLMResponse:
    """Call the LLM and record a full trace in Langfuse.

    Records:
    - A retrieval span with all returned chunks and the sufficiency flag
    - An LLM generation span with prompt, response, and token counts
    """
    langfuse = get_langfuse()
    llm = get_llm_provider()

    # Determine the model name for display in Langfuse
    model_name = (
        settings.openrouter.llm_model
        if settings.llm_provider == "openrouter"
        else settings.bedrock.llm_model
    )

    trace = langfuse.trace(
        name="rag-query",
        user_id=user_id,
        input={"question": question},
    )

    span = trace.span(name="retrieval", input={"question": question})
    span.end(output={
        "chunks": [vars(c) for c in retrieval_result.chunks],
        "sufficient": retrieval_result.sufficient,
    })

    generation = trace.generation(
        name="llm-complete",
        model=model_name,
        input=messages,
    )

    response = await llm.complete(messages)

    generation.end(
        output=response.text,
        usage={
            "promptTokens": response.input_tokens,
            "completionTokens": response.output_tokens,
        },
    )

    trace.update(output={"answer": response.text})

    return response
