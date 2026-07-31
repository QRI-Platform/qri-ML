import sys
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import HTTPException
from src.core.logger import logger
from src.core.exceptions import MyException
from api.middlewares.multi_middleware import multer_middleware
from api.schemas.chat_schema import ChatRequest
from api.middlewares.authentication_middleware import authenticate_user
from src.pipelines.graph_runner_pipeline import get_graph_runner_pipeline


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


@router.post("/ingest")
async def ingest_vec_data(request: Request, file_path: str = Depends(multer_middleware)):
    try:
        logger.info("ingest endpoint: user=%s thread=%s file=%s",
                    request.state.user_id, request.state.thread_id, file_path)
        pipeline = get_graph_runner_pipeline()
        async for _ in pipeline.initiate(
            user_id=request.state.user_id,
            thread_id=request.state.thread_id,
            file_paths=[file_path] if file_path else [],
        ):
            pass
        logger.info("ingest endpoint: completed for thread=%s", request.state.thread_id)
        return JSONResponse(content={"success": True, "message": "Data ingested successfully", "data": None}, status_code=200)
    except Exception as e:
        logger.error("ingest endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.post("/chat")
async def run_workflow(request: Request, payload: ChatRequest):
    try:
        logger.info("chat endpoint: user=%s thread=%s", request.state.user_id, request.state.thread_id)
        return StreamingResponse(
            stream_chat(payload.message, request.state.user_id, request.state.thread_id),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error("chat endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})
