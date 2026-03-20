import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from mangum import Mangum

from src.api.deps import get_current_user
from src.api.routes.documents import router as documents_router
from src.api.routes.evaluate import router as evaluate_router
from src.api.routes.query import router as query_router
from src.auth import UserContext
from src.config import settings
from src.vector_store import ensure_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG anything",
    version="0.1.0",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "clientId": settings.authentik.client_id,
        "scopes": "openid email profile",
        "usePkceWithAuthorizationCodeGrant": True,
    },
)

# ---------------------------------------------------------------------------
# CORS — restrict to known origins; never use allow_origins=["*"] with credentials
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(query_router, tags=["query"])
app.include_router(evaluate_router, tags=["evaluate"])


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "environment": settings.environment, "version": "0.1.0"}


from fastapi import Depends  # noqa: E402 — keep below app init


@app.get("/me", tags=["auth"])
async def get_me(current_user: UserContext = Depends(get_current_user)):
    return {"user_id": current_user.user_id, "email": current_user.email}


# Lambda handler
handler = Mangum(app)
