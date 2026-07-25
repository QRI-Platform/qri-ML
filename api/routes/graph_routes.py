import sys
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from src.logger import logger
from src.exception import MyException
from api.middlewares.multi_middleware import multer_middleware
from api.models.chat_model import ChatRequest
from api.middlewares.authentication_middleware import authenticate_user
from src.pipelines import get_pipeline
from api.helper.graph_helper import _delete_pinecone_namespace
from src.core.dependencies import get_thread_manager

router = APIRouter(dependencies=[Depends(authenticate_user)]) 


async def stream_chat(message: str, user_id: str, thread_id: str):
    try:
        logger.info("stream_chat started: user=%s thread=%s", user_id, thread_id)
        pipeline = get_pipeline()
        async for event in pipeline.initiate(user_id=user_id, thread_id=thread_id, message=message):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield chunk.content
        logger.info("stream_chat completed: user=%s thread=%s", user_id, thread_id)
    except Exception as e:
        logger.error("stream_chat error: %s", str(e))
        raise MyException(e, sys)


@router.post("/ingest")
async def ingest_vec_data(request: Request, file_path: str = Depends(multer_middleware)):
    try:
        logger.info("ingest endpoint: user=%s thread=%s file=%s",
                    request.state.user_id, request.state.thread_id, file_path)
        pipeline = get_pipeline()
        async for _ in pipeline.initiate(
            user_id=request.state.user_id,
            thread_id=request.state.thread_id,
            file_paths=[file_path] if file_path else [],
        ):
            pass
        logger.info("ingest endpoint: completed for thread=%s", request.state.thread_id)
        return {"success": True, "message": "Data ingested successfully"}
    except Exception as e:
        logger.error("ingest endpoint failed: %s", str(e))
        raise MyException(e, sys)


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
        raise MyException(e, sys)


@router.delete("/delete")
async def delete_thread_endpoint(request: Request):
    try:
        logger.info("delete endpoint: user=%s thread=%s", request.state.user_id, request.state.thread_id)
        get_thread_manager().remove_thread(request.state.thread_id)
        await _delete_pinecone_namespace(request.state.thread_id)
        logger.info("delete endpoint: completed for thread=%s", request.state.thread_id)
        return {"success": True, "message": "Thread deleted successfully"}
    except Exception as e:
        logger.error("delete endpoint failed: %s", str(e))
        raise MyException(e, sys)