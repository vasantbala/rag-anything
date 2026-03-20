from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.api.models import QueryRequest, QueryResponse
from src.auth import UserContext
from src.rag_pipeline import answer_question

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
