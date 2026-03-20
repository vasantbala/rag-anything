"""Offline RAG evaluation using RAGAS metrics.

This module is intentionally on-demand: evaluation costs LLM tokens and should
only be called from an explicit endpoint, not on every user query.

Both Faithfulness and AnswerRelevancy are computed against OpenRouter so that
evaluation works in dev without AWS credentials.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from src.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_ragas_llm() -> LangchainLLMWrapper:
    lc_llm = ChatOpenAI(
        model=settings.openrouter.llm_model,
        api_key=settings.openrouter.api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )
    return LangchainLLMWrapper(lc_llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    lc_embeddings = OpenAIEmbeddings(
        model=settings.openrouter.embedding_model,
        api_key=settings.openrouter.api_key,
        base_url=OPENROUTER_BASE_URL,
    )
    return LangchainEmbeddingsWrapper(lc_embeddings)


def evaluate_rag(question: str, answer: str, contexts: list[str]) -> dict:
    """Compute Faithfulness and AnswerRelevancy scores for one RAG response.

    Args:
        question: The user question.
        answer: The LLM's answer.
        contexts: The retrieved text chunks that were fed to the LLM.

    Returns:
        {"faithfulness": float, "answer_relevancy": float}
    """
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    result = evaluate(dataset, metrics=metrics)

    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }
