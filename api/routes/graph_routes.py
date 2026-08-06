import sys
from typing import List
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import HTTPException
from src.core.logger import logger
from src.core.exceptions import MyException
from api.middlewares.multi_middleware import multer_middleware
from api.schemas.chat_schema import ChatRequest
from api.middlewares.authentication_middleware import authenticate_user
from src.pipelines.graph_runner_pipeline import get_graph_runner_pipeline
from typing import Any

router = APIRouter(dependencies=[Depends(authenticate_user)])


async def stream_chat(message: str, user_id: str, thread_id: str):
    try:
        logger.info("stream_chat started: user=%s thread=%s", user_id, thread_id)
        pipeline = get_graph_runner_pipeline()
        streamed_any = False
        async for event in pipeline.initiate(user_id=user_id, thread_id=thread_id, message=message):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and getattr(chunk, "content", None):
                    streamed_any = True
                    yield str(chunk.content)
            elif event.get("event") == "on_chain_end" and event.get("name") == "chat_node":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output.get("ai_response") and not streamed_any:
                    yield str(output["ai_response"])
        logger.info("stream_chat completed: user=%s thread=%s", user_id, thread_id)
    except Exception as e:
        logger.error("stream_chat error: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.post(
    "/ingest",
    summary="Ingest Documents into Pinecone",
    responses={
        200: {"description": "All files processed and vectors upserted successfully."},
        400: {"description": "File processing or Pinecone upsert failed."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def ingest_vec_data(request: Request, file_paths: List[str] = Depends(multer_middleware)):
    """
    Upload one or more documents to be parsed, chunked, and stored in the user's
    dedicated Pinecone namespace (`thread_id`).

    - **Supported formats:** PDF, DOCX, TXT and any format supported by Docling.
    - **Multimodal:** Docling extracts text, tables (as markdown), and OCR content
      from images — all indexed as searchable chunks.
    - **Filename tagging:** Each chunk is tagged with `{thread_id}_{filename}` in
      metadata, enabling `@filename` scoped retrieval in the chat endpoint.
    - **Parallel loading:** Multiple files are processed concurrently.

    ### Authentication
    Pass `user_id` and `thread_id` as query params or headers (`x-user-id`, `x-thread-id`).

    ### Request
    `multipart/form-data` — field name: `files` (repeat for multiple files).
    """
    try:
        logger.info("ingest endpoint: user=%s thread=%s files=%d",
                    request.state.user_id, request.state.thread_id, len(file_paths))
        pipeline = get_graph_runner_pipeline()
        async for _ in pipeline.initiate(
            user_id=request.state.user_id,
            thread_id=request.state.thread_id,
            file_paths=file_paths,
        ):
            pass
        logger.info("ingest endpoint: completed for thread=%s", request.state.thread_id)
        return JSONResponse(content={"success": True, "message": "Data ingested successfully", "data": None}, status_code=200)
    except Exception as e:
        logger.error("ingest endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.post(
    "/chat",
    summary="Chat with the RAG Pipeline (SSE Stream)",
    responses={
        200: {"description": "Server-Sent Events stream of AI response tokens."},
        400: {"description": "Pipeline execution error."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def run_workflow(request: Request, payload: ChatRequest):
    """
    Send a message to the stateful LangGraph RAG pipeline and receive a
    **Server-Sent Events (SSE)** stream of AI response tokens.

    ### Pipeline Flow
    1. **Orchestrator** — decides if Pinecone retrieval is needed.
    2. **Query Generation** — expands the user query into multiple search queries.
    3. **Retriever** — fetches relevant chunks from Pinecone (parallel search).
    4. **Chat Node** — Groq LLM generates a grounded, streaming response.

    ### `@filename` Scoped Retrieval
    Mention `@report.pdf` in your message to restrict retrieval to only that
    file's chunks. Works for any file previously ingested in this thread.

    ### Long-Term Memory
    The LLM automatically extracts and persists key user facts across sessions.

    ### Authentication
    Pass `user_id` and `thread_id` as query params or headers (`x-user-id`, `x-thread-id`).

    ### Response
    `Content-Type: text/event-stream` — consume as SSE or read chunks directly.
    """
    try:
        logger.info("chat endpoint: user=%s thread=%s", request.state.user_id, request.state.thread_id)
        return StreamingResponse(
            stream_chat(payload.message, request.state.user_id, request.state.thread_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error("chat endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})

