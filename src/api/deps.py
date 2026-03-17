from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer, HTTPAuthorizationCredentials
from src.auth import verify_jwt, UserContext
from src.config import settings

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=settings.authentik.authorization_url,
    tokenUrl=settings.authentik.token_url,
)

async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)
) -> UserContext:
    return await verify_jwt(credentials)