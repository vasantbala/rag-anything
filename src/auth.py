
from dataclasses import dataclass 
from typing import Any

import httpx
from cachetools import TTLCache
from fastapi import HTTPException, status
from jose import jwt, JWTError

from src.config import settings

_jwks_cache: TTLCache = TTLCache(maxsize=1, ttl=600)

@dataclass
class UserContext:
    user_id: str # the "sub" claim from the JWT
    email: str # the "email" claim from the JWT

async def fetch_jwks() -> dict[str, Any]:
    if "jwks" in _jwks_cache:
        return _jwks_cache["jwks"]
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(settings.authentik.jwks_url, timeout=10)
            response.raise_for_status()
            jwks = response.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch JWKS: {e}"
        ) 
    _jwks_cache["jwks"] = jwks
    return jwks
    

async def verify_jwt(token: str) -> UserContext:
    jwks = await fetch_jwks()
    try:
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        print("JWT payload:", payload)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}"
        )

    user_id: str | None = payload.get("sub")
    email: str | None = payload.get("email")

    if not user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required claims"
        )

    return UserContext(user_id=user_id, email=email)