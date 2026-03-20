"""Tests for the LLM abstraction layer and RAG pipeline short-circuit logic."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.base import LLMResponse
from src.llm.openai import OpenAICompatibleProvider
from src.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is RAG?"},
]

FAKE_RESPONSE = LLMResponse(
    text="RAG stands for Retrieval-Augmented Generation.",
    input_tokens=20,
    output_tokens=10,
)


def _openai_response(text: str, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    """Build a mock object shaped like openai.types.chat.ChatCompletion."""
    choice = MagicMock()
    choice.message.content = text

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# 1. OpenAI-compatible provider (used by OpenRouter)
# ---------------------------------------------------------------------------

async def test_openai_compatible_provider_parses_response():
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test-model",
    )

    fake = _openai_response("RAG stands for Retrieval-Augmented Generation.", 20, 10)

    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake),
    ):
        result = await provider.complete(MESSAGES)

    assert result.text == "RAG stands for Retrieval-Augmented Generation."
    assert result.input_tokens == 20
    assert result.output_tokens == 10


async def test_openai_compatible_provider_passes_correct_params():
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="my-model",
    )

    fake = _openai_response("answer", 5, 3)
    create_mock = AsyncMock(return_value=fake)

    with patch.object(provider._client.chat.completions, "create", new=create_mock):
        await provider.complete(MESSAGES)

    create_mock.assert_called_once_with(
        model="my-model",
        messages=MESSAGES,
        max_tokens=1024,
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# 2. OpenRouterProvider — wraps OpenAICompatibleProvider with settings
# ---------------------------------------------------------------------------

async def test_openrouter_provider_uses_settings():
    mock_settings = MagicMock()
    mock_settings.openrouter.api_key = "or-key"
    mock_settings.openrouter.llm_model = "meta-llama/llama-3-8b-instruct"

    with patch("src.llm.openrouter.settings", mock_settings):
        from src.llm.openrouter import OpenRouterProvider
        provider = OpenRouterProvider()

    assert provider.model == "meta-llama/llama-3-8b-instruct"


# ---------------------------------------------------------------------------
# 3. BedrockProvider — sync boto3 wrapped in asyncio.to_thread
# ---------------------------------------------------------------------------

async def test_bedrock_provider_parses_response():
    mock_settings = MagicMock()
    mock_settings.bedrock.region = "us-east-1"
    mock_settings.bedrock.llm_model = "amazon.nova-micro-v1:0"
    mock_settings.aws_access_key_id = "AKID"
    mock_settings.aws_secret_access_key = "secret"

    fake_bedrock_response = {
        "output": {"message": {"content": [{"text": "A RAG system retrieves documents."}]}},
        "usage": {"inputTokens": 15, "outputTokens": 8},
    }

    with (
        patch("src.llm.bedrock.settings", mock_settings),
        patch("boto3.client") as mock_boto,
    ):
        mock_client = MagicMock()
        mock_client.converse.return_value = fake_bedrock_response
        mock_boto.return_value = mock_client

        from src.llm.bedrock import BedrockProvider
        provider = BedrockProvider()
        result = await provider.complete(MESSAGES)

    assert result.text == "A RAG system retrieves documents."
    assert result.input_tokens == 15
    assert result.output_tokens == 8


async def test_bedrock_provider_separates_system_messages():
    """System messages must be passed in the 'system' field, not conversation."""
    mock_settings = MagicMock()
    mock_settings.bedrock.region = "us-east-1"
    mock_settings.bedrock.llm_model = "amazon.nova-micro-v1:0"
    mock_settings.aws_access_key_id = "AKID"
    mock_settings.aws_secret_access_key = "secret"

    fake_response = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 5, "outputTokens": 2},
    }

    with (
        patch("src.llm.bedrock.settings", mock_settings),
        patch("boto3.client") as mock_boto,
    ):
        mock_client = MagicMock()
        mock_client.converse.return_value = fake_response
        mock_boto.return_value = mock_client

        from src.llm.bedrock import BedrockProvider
        provider = BedrockProvider()
        await provider.complete(MESSAGES)

    call_kwargs = mock_client.converse.call_args[1]
    # System prompt extracted correctly
    assert call_kwargs["system"] == [{"text": "You are helpful."}]
    # Only user message in conversation
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# 4. Factory
# ---------------------------------------------------------------------------

def test_factory_returns_openrouter_provider():
    mock_settings = MagicMock()
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter.api_key = "key"
    mock_settings.openrouter.llm_model = "model"

    with (
        patch("src.llm.factory.settings", mock_settings),
        patch("src.llm.openrouter.settings", mock_settings),
    ):
        from src.llm.factory import get_llm_provider
        get_llm_provider.cache_clear()
        provider = get_llm_provider()

    from src.llm.openrouter import OpenRouterProvider
    assert isinstance(provider, OpenRouterProvider)
    get_llm_provider.cache_clear()


def test_factory_returns_bedrock_provider():
    mock_settings = MagicMock()
    mock_settings.llm_provider = "bedrock"
    mock_settings.bedrock.region = "us-east-1"
    mock_settings.bedrock.llm_model = "amazon.nova-micro-v1:0"
    mock_settings.aws_access_key_id = "AKID"
    mock_settings.aws_secret_access_key = "secret"

    with (
        patch("src.llm.factory.settings", mock_settings),
        patch("src.llm.bedrock.settings", mock_settings),
        patch("boto3.client"),
    ):
        from src.llm.factory import get_llm_provider
        get_llm_provider.cache_clear()
        provider = get_llm_provider()

    from src.llm.bedrock import BedrockProvider
    assert isinstance(provider, BedrockProvider)
    get_llm_provider.cache_clear()


def test_factory_raises_on_unknown_provider():
    mock_settings = MagicMock()
    mock_settings.llm_provider = "unknown-ai"

    with patch("src.llm.factory.settings", mock_settings):
        from src.llm.factory import get_llm_provider
        get_llm_provider.cache_clear()
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_provider()
        get_llm_provider.cache_clear()


# ---------------------------------------------------------------------------
# 5. answer_question — low confidence short-circuit
# ---------------------------------------------------------------------------

async def test_answer_question_short_circuits_when_insufficient():
    """When retrieval returns sufficient=False, no LLM call should be made."""
    insufficient_result = RetrievalResult(chunks=[], sufficient=False)

    with (
        patch("src.rag_pipeline.retrieve", new=AsyncMock(return_value=insufficient_result)),
        patch("src.rag_pipeline.traced_llm_call", new=AsyncMock()) as mock_llm,
    ):
        from src.rag_pipeline import answer_question
        result = await answer_question("What is RAG?", user_id="test-user")

    mock_llm.assert_not_called()
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
    assert result["confidence"] == 0.0


async def test_answer_question_returns_answer_and_sources():
    from src.retriever import RetrievedChunk

    chunk = RetrievedChunk(
        text="RAG uses retrieval.",
        doc_id="doc-1",
        source_type="pdf",
        chunk_index=0,
        page_number=1,
        reranker_score=0.85,
    )
    sufficient_result = RetrievalResult(chunks=[chunk], sufficient=True)

    with (
        patch("src.rag_pipeline.retrieve", new=AsyncMock(return_value=sufficient_result)),
        patch(
            "src.rag_pipeline.traced_llm_call",
            new=AsyncMock(return_value=FAKE_RESPONSE),
        ),
    ):
        from src.rag_pipeline import answer_question
        result = await answer_question("What is RAG?", user_id="test-user")

    assert result["answer"] == FAKE_RESPONSE.text
    assert len(result["sources"]) == 1
    assert result["sources"][0]["doc_id"] == "doc-1"
    assert result["confidence"] == pytest.approx(0.85)
