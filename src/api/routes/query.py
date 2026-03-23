from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.api.models import QueryRequest, QueryResponse, RetrieveResponse, RetrievedChunkModel
from src.auth import UserContext
from src.rag_pipeline import answer_question
from src.retriever import retrieve

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: UserContext = Depends(get_current_user),
) -> QueryResponse:
    result = await answer_question(
        question=request.question,
        user_id=current_user.user_id,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
    )
    return QueryResponse(**result)


# --- New: /retrieve endpoint ---
@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(
    request: QueryRequest,
    current_user: UserContext = Depends(get_current_user),
) -> RetrieveResponse:
    result = await retrieve(
        question=request.question,
        user_id=current_user.user_id,
        doc_ids=request.doc_ids,
        top_k=request.top_k,
    )
    # Convert dataclass chunks to Pydantic models
    chunk_models = [RetrievedChunkModel(**c.__dict__) for c in result.chunks]
    return RetrieveResponse(chunks=chunk_models, sufficient=result.sufficient)
