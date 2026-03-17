import pytest
from unittest.mock import AsyncMock, patch

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from src.api.main import app
from src.auth import verify_jwt

# ── RSA keypair generated once for the module ────────────────────────────────
_private_key = rsa.generate_private_key(
    public_exponent=65537, key_size=2048, backend=default_backend()
)
_PRIVATE_PEM = _private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
_PUBLIC_PEM = _private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()


def _make_token(claims: dict) -> str:
    return jwt.encode(claims, _PRIVATE_PEM, algorithm="RS256")


# ── Test 1 — valid token ──────────────────────────────────────────────────────
async def test_verify_jwt_valid_token():
    claims = {"sub": "user-123", "email": "test@example.com"}
    token = _make_token(claims)

    with patch("src.auth.fetch_jwks", new=AsyncMock(return_value=_PUBLIC_PEM)):
        ctx = await verify_jwt(token)

    assert ctx.user_id == "user-123"
    assert ctx.email == "test@example.com"


# ── Test 2 — invalid / expired token ─────────────────────────────────────────
async def test_verify_jwt_invalid_token():
    with patch("src.auth.fetch_jwks", new=AsyncMock(return_value=_PUBLIC_PEM)):
        with pytest.raises(HTTPException) as exc_info:
            await verify_jwt("this.is.garbage")

    assert exc_info.value.status_code == 401


# ── Test 3 — missing Authorization header ────────────────────────────────────
def test_me_missing_auth_header():
    client = TestClient(app)
    response = client.get("/me")
    assert response.status_code == 401
