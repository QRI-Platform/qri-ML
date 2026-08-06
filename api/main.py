from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.logger import logger
from src.core.dependencies import warmup_dependencies
from src.pipelines.graph_runner_pipeline import get_graph_runner_pipeline
from src.core.dependencies import connection_pool, close_connection_pool
from api.routes.graph_routes import router as graph_router
from api.routes.user_routes import router as UserRouter


tags_metadata = [
    {
        "name": "RAG Pipeline",
        "description": (
            "Core LangGraph-powered RAG (Retrieval-Augmented Generation) endpoints. "
            "Use **`/ingest`** to upload and index documents into Pinecone, "
            "and **`/chat`** to stream AI responses grounded in those documents."
        ),
    },
    {
        "name": "User & Conversation",
        "description": (
            "Endpoints for managing per-user conversation history stored in the "
            "LangGraph checkpointer, and long-term memory persisted in the BaseStore."
        ),
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server startup: warming up all singletons...")
    await connection_pool()       # Must run first — initializes _checkpointer & _store
    warmup_dependencies()
    get_graph_runner_pipeline()
    logger.info("All singletons initialized — server is ready")
    yield
    logger.info("Server shutdown")
    await close_connection_pool()


app = FastAPI(
    lifespan=lifespan,
    title="QRI — Multi-Tenant Stateful RAG API",
    description=(
        "An industrial-grade, stateful Multi-Tenant RAG backend built with "
        "**FastAPI**, **LangGraph**, **Pinecone**, and **Groq LLM**.\n\n"
        "### Authentication\n"
        "Every endpoint requires `user_id` and `thread_id` — passed as **query params** "
        "(`?user_id=...&thread_id=...`) or **headers** (`x-user-id`, `x-thread-id`).\n\n"
        "### `@filename` Filter\n"
        "In the chat message, mention `@filename.pdf` to restrict retrieval to only "
        "that file's chunks in Pinecone."
    ),
    version="1.0.0",
    contact={"name": "VashuTheGreat"},
    openapi_tags=tags_metadata,
)
app.include_router(graph_router, prefix="/api/v1/graph", tags=["RAG Pipeline"])
app.include_router(UserRouter, prefix="/api/v1/user", tags=["User & Conversation"])
