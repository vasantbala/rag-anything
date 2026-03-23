import re

from pydantic import BaseModel, Field, field_validator

_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
_ALLOWED_DB_SCHEMES = ("postgresql://", "mysql://")

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DbSourceRequest(BaseModel):
    connection_string: str
    table_name: str = Field(..., description="Alphanumeric + underscore, max 64 chars")
    content_columns: list[str] = Field(..., min_length=1)
    metadata_columns: list[str] = []
    row_id_column: str

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        if not any(v.startswith(scheme) for scheme in _ALLOWED_DB_SCHEMES):
            raise ValueError("connection_string must start with postgresql:// or mysql://")
        return v

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        if not _TABLE_NAME_RE.match(v):
            raise ValueError(
                "table_name must match ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$ "
                "(alphanumeric and underscores only, max 64 chars)"
            )
        return v


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    doc_id: str
    name: str
    source_type: str
    status: str
    chunk_count: int
    created_at: str


class IngestResponse(BaseModel):
    doc_id: str
    status: str



# --- Retrieval-only response (no LLM) ---
class RetrievedChunkModel(BaseModel):
    text: str
    doc_id: str
    source_type: str
    chunk_index: int
    page_number: int | None = None
    reranker_score: float
    metadata: dict = {}

class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunkModel]
    sufficient: bool

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float


class EvaluateResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
