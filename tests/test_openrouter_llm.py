"""Integration test for OpenRouter LLM — run on demand, not in CI.

Usage:
    poetry run pytest tests/test_openrouter_llm.py -v -s

Requires a valid OPENROUTER__API_KEY and OPENROUTER__LLM_MODEL in .env.
"""
import pytest

from src.config import settings
from src.llm.openrouter import OpenRouterProvider

pytestmark = pytest.mark.integration  # skip in CI unless -m integration is passed


async def test_openrouter_llm_basic_call():
    """Send a minimal message and verify we get a non-empty text response."""
    provider = OpenRouterProvider()
    print(f"\nModel: {provider.model}")

    messages = [
        {"role": "user", "content": "Reply with exactly: OK"},
    ]
    response = await provider.complete(messages)

    print(f"Response text: {response.text!r}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")

    assert response.text, "Expected non-empty response text"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


async def test_openrouter_llm_model_name():
    """Print the configured model so we can confirm it at runtime."""
    print(f"\nConfigured LLM model: {settings.openrouter.llm_model!r}")
    print(f"Configured embedding model: {settings.openrouter.embedding_model!r}")
    # Not a hard assertion — just visibility
    assert settings.openrouter.llm_model, "OPENROUTER__LLM_MODEL is not set in .env"


async def test_openrouter_llm_rag_style_call():
    """Test with a system+user message pair matching real RAG usage."""
    provider = OpenRouterProvider()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using only the context below.\n\n"
                "Context:\n"
                "[Source 1: doc_id=test-doc, page=1]\n"
                "The SOLID principles are five design principles in object-oriented programming: "
                "Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, "
                "and Dependency Inversion."
            ),
        },
        {"role": "user", "content": "What are the SOLID principles?"},
    ]

    response = await provider.complete(messages)

    print(f"\nModel: {provider.model}")
    print(f"Answer: {response.text}")
    print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")

    assert response.text
    assert any(
        word in response.text.lower()
        for word in ("solid", "single", "responsibility", "open", "liskov")
    ), f"Expected SOLID-related content in response, got: {response.text!r}"
