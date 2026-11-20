
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.routes import router
from app.core.config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create required storage directories when the app starts
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    print("Smart Research Agent is starting...")

    yield

    print("Smart Research Agent is shutting down...")


app = FastAPI(
    title=settings.app_name,
    description="A local RAG and tool-using research agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Smart Research Agent is running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "llm_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
    }
