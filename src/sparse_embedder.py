from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

"""
1. Model is downloaded at cold start in Lambda
The ~50MB BM25 model is downloaded on first import. In Lambda this means the first cold start after a new deployment can time out if your function timeout is short. Production fix: bake the model into the Docker image at build time by adding a one-line pre-download step in the Dockerfile:

2. No batching limit
Unlike Bedrock, FastEmbed runs in-process, so there's no API rate limit. However, passing thousands of texts at once will spike memory. Production code processes in batches of ~100 to keep memory usage flat, especially important inside Lambda which has a fixed memory ceiling.

3. BM25 isn't aware of your domain vocabulary
The standard Qdrant/bm25 model uses a generic English vocabulary. For domain-specific corpora (medical, legal, code), rare terms get low IDF weights or are out-of-vocabulary entirely. Production systems fine-tune the BM25 vocabulary on their own corpus using fastembed's custom model path. For this project the generic model is fine.

4. No fallback if fastembed crashes
If FastEmbed's model file is corrupted on disk, embed_sparse raises an unhandled exception that kills the entire ingestion. A try/except that logs the error and re-raises a clean RuntimeError("Sparse embedding failed") makes the failure easier to diagnose in CloudWatch.
"""


# Instantiated once at module level — downloads the model on first use (~50MB, cached to disk)
_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def embed_sparse(texts: list[str]) -> list[SparseVector]:
    """
    BM25 is sensitive to casing and punctuation. 
    Production pipelines lowercasee and strip punctuation before BM25 embedding to improve term matching. 
    Dense embeddings don't need this (the model handles it internally), but BM25 is pure keyword frequency — "FastAPI" 
    and "fastapi" are treated as different terms without normalization.
    """
    normalized = [t.lower() for t in texts]
    embeddings = list(_model.embed(normalized))
    return [
        SparseVector(
            indices=emb.indices.tolist(), # numpy array → plain Python list
            values=emb.values.tolist(),
        )
        for emb in embeddings
    ]