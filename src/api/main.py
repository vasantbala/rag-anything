from fastapi import FastAPI
from mangum import Mangum
from src.config import settings


app = FastAPI(title="RAG anything", version="0.0.1")

@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.environment, "version": "0.0.1"}

# Lambda handler
handler = Mangum(app)