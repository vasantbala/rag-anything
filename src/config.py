from pydantic_settings import BaseSettings, SettingsConfigDict

class QdrantSettings(BaseSettings):
    url: str
    api_key: str
    collection_name: str

class LangFuseSettings(BaseSettings):
    public_key: str
    secret_key: str
    host: str    

class AuthentikSettings(BaseSettings):
    jwks_url: str
    client_id: str
    authorization_url: str
    token_url: str

class OpenRouterSettings(BaseSettings):
    api_key: str
    llm_model: str

class BedrockSettings(BaseSettings):
    region: str
    embedding_model: str
    llm_model: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")
    environment: str
    chunk_size: int
    chunk_overlap: int    
    retrieval_top_k: int
    reranker_score_threshold: float
    llm_provider: str
    s3_bucket_name: str
    dynamodb_table_name: str
    qdrant: QdrantSettings
    authentik: AuthentikSettings
    bedrock: BedrockSettings
    langfuse: LangFuseSettings
    openrouter: OpenRouterSettings
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB default
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

settings = Settings()
