from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from mangum import Mangum
from src.config import settings
from src.api.deps import get_current_user
from src.auth import UserContext
from src.vector_store import ensure_collection
from fastapi.security import OAuth2AuthorizationCodeBearer


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG anything", 
    version="0.0.1",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "clientId": settings.authentik.client_id,
        "scopes": "openid email profile",
        "usePkceWithAuthorizationCodeGrant": True
    })

@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.environment, "version": "0.0.1"}

@app.get("/me")
async def get_me(current_user: UserContext = Depends(get_current_user)):
    return {"user_id": current_user.user_id, "email": current_user.email}

# Lambda handler
handler = Mangum(app)