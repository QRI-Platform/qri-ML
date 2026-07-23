import sys
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from api.middlewares.multi_middleware import multer_middleware
from api.middlewares.authentication_middleware import authenticate_user
from src.graphs.builder import graph
from src.models.workflow_models import State
from src.exception import MyException

router = APIRouter(dependencies=[Depends(authenticate_user)])


async def stream_chat(message: str, thread_id: str):
    state = State(user_id=thread_id, messages=[HumanMessage(content=message)])
    async for event in graph.astream_events(state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield chunk.content


@router.post("/ingest")
async def ingest_vec_data(request: Request, file_path: str = Depends(multer_middleware)):
    try:
        state = State(user_id=request.state.thread_id, file_paths=[file_path])
        await graph.ainvoke(state)
        return {"success": True, "message": "Data ingested successfully"}
    except Exception as e:
        raise MyException(e, sys)


@router.post("/chat")
async def run_workflow(request: Request, message: str = Form(...)):
    try:
        return StreamingResponse(
            stream_chat(message, request.state.thread_id),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise MyException(e, sys)