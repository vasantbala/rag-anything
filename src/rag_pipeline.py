from src.observability import traced_llm_call
from src.prompt_builder import build_prompt
from src.retriever import retrieve


async def answer_question(
    question: str,
    user_id: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict:
    """Run the full RAG pipeline: retrieve → build prompt → call LLM → return answer.

    Returns a dict with:
      - answer: str
      - sources: list of dicts with doc_id, chunk_index, page_number, score
      - confidence: reranker score of the top chunk (0.0 if no chunks)
    """
    retrieval_result = await retrieve(question, user_id, doc_ids, top_k)

    if not retrieval_result.sufficient:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
            "confidence": 0.0,
        }

    messages = build_prompt(question, retrieval_result.chunks)
    response = await traced_llm_call(question, messages, retrieval_result, user_id)

    sources = [
        {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "score": c.reranker_score,
        }
        for c in retrieval_result.chunks
    ]

    return {
        "answer": response.text,
        "sources": sources,
        "confidence": retrieval_result.chunks[0].reranker_score if retrieval_result.chunks else 0.0,
    }
