from fastapi import APIRouter, Request,Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from src.core.logger import logger
from api.middlewares.authentication_middleware import authenticate_user
from src.services.conversation_service import (
    load_conversation,
    delete_pinecone_namespace,
    get_user_long_term_memory,
    delete_user_conversation,
)

router: APIRouter = APIRouter(dependencies=[Depends(authenticate_user)])


@router.get("/conversation")
async def get_user_conversation(request:Request):
    try:
        user_id: str = request.state.user_id
        thread_id: str = request.state.thread_id
        logger.info("Retrieving user conversation")
        messages = await load_conversation(thread_id=thread_id, user_id=user_id)
        return JSONResponse(content={"data": messages, "message": "retrieved data successfully", "success": True})
    except Exception as e:
        logger.error("Error while loading user conversations: %s", str(e))
        raise HTTPException(status_code=400, detail={"data": None, "messages": str(e), "success": False})


@router.get("/long_term_memory/")
async def get_users_long_term_memory(request:Request):
    try:
        logger.info("Retrieving user long term memory for %s", request.state.user_id)
        messages = await get_user_long_term_memory(user_id=request.state.user_id)
        return JSONResponse(content={"data": messages, "message": "retrieved long term memory successfully", "success": True})
    except Exception as e:
        logger.error("Error while retrieving long term memory: %s", str(e))
        raise HTTPException(status_code=400, detail={"data": None, "messages": str(e), "success": False})


@router.delete("/pine_cone")
async def delete_thread_endpoint(request: Request):
    try:
        logger.info("delete endpoint: user=%s thread=%s", request.state.user_id, request.state.thread_id)
        await delete_pinecone_namespace(request.state.thread_id)
        logger.info("delete endpoint: completed for thread=%s", request.state.thread_id)
        return JSONResponse(content={"success": True, "message": "Thread deleted successfully", "data": None}, status_code=200)
    except Exception as e:
        logger.error("delete endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.delete("/conversation")
async def delete_user_conversation_endpoint(request: Request):
    try:
        success = await delete_user_conversation(thread_id=request.state.thread_id, user_id=request.state.user_id)
        if success:
            return JSONResponse(content={"success": success, "message": "Conversation deleted", "data": None})
        
        else:
            return JSONResponse(content={"success": success, "message": "Conversation didn't deleted", "data": None})
    except Exception as e:
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})
