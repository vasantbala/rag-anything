from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

"""
Here's what's missing at true production scale:

1. Character-based vs. token-based splitting
RecursiveCharacterTextSplitter counts characters, not tokens. A chunk of 512 characters might be 150 tokens or 400 tokens depending on the language and content.
At production scale you'd use tiktoken to count actual tokens and ensure chunks never exceed the embedding model's limit. Bedrock Titan v2 has a hard cap of 8192 tokens — the current code could silently hit that with dense text.

2. No streaming / generator pipeline
The current code loads everything into memory before chunking, then holds all chunks before embedding. 
For a 500-page PDF this is fine; for a 10,000-page technical manual it becomes a memory problem. Production pipelines use generators and process in a streaming fashion — emit one chunk at a time rather than collecting all into a list.

3. No parent-child chunking
A popular production technique: store small chunks (128 tokens) for precise retrieval, but also store a parent chunk (512 tokens) in the payload. 
At retrieval time you fetch the small chunk (high precision) but return the parent context to the LLM (more information). The current design only has one granularity.

4. No deduplication
If someone uploads the same document twice, it re-ingests. The deterministic UUIDs in vector_store.py handle the Qdrant side (upsert overwrites), but DynamoDB would get a duplicate entry. A content hash check before ingestion would catch this.

5. No async / concurrent embedding
embed_texts calls Bedrock synchronously in batches of 10. At scale you'd call multiple batches concurrently with asyncio.gather.
"""
def chunk_documents(pages: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    # from_tiktoken_encoder counts actual tokens, not characters
    # cl100k_base is the BPE tokenizer used by GPT-4 / text-embedding-3 —
    # Titan v2 doesn't publish its tokenizer so this is the industry-standard proxy
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    results = []
    for text, metadata in pages:
        chunks = splitter.split_text(text)
        for chunk_index, chunk_text in enumerate(chunks):
            results.append((chunk_text, {**metadata, "chunk_index": chunk_index}))

    return results