import asyncio

from fastapi import APIRouter, Depends

from src.api.deps import get_current_user
from src.api.models import EvaluateRequest, EvaluateResponse
from src.auth import UserContext
from src.evaluation import evaluate_rag

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluateRequest,
    current_user: UserContext = Depends(get_current_user),  # noqa: ARG001 — auth required
) -> EvaluateResponse:
    # evaluate_rag is sync and makes network calls; run off the event loop
    scores = await asyncio.to_thread(
        evaluate_rag,
        request.question,
        request.answer,
        request.contexts,
    )
    return EvaluateResponse(**scores)
