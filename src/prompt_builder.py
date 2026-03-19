from src.retriever import RetrievedChunk


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i + 1}: doc_id={chunk.doc_id}, page={chunk.page_number}]\n{chunk.text}"
        )
    context = "\n\n".join(context_parts)

    system_message = (
        "You are a helpful assistant. Answer the user's question using ONLY the context provided below.\n"
        "If the context does not contain enough information to answer, say "
        '"I don\'t have enough information to answer that."\n'
        "Do not make up facts. Cite the source document name when relevant.\n\n"
        f"Context:\n{context}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]