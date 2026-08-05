from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from src.core.logger import logger
from api.middlewares.authentication_middleware import authenticate_user
from src.services.conversation_service import (
    load_conversation,
    delete_pinecone_namespace,
    get_user_long_term_memory,
    delete_user_conversation,
    delete_long_term_memory_key,
    upsert_long_term_memory,
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


# ── Long-Term Memory Management ──────────────────────────────────────────────

class UpsertMemoryRequest(BaseModel):
    key: str
    value: str


@router.post("/long_term_memory")
async def add_or_update_long_term_memory(request: Request, body: UpsertMemoryRequest):
    """Add or update a specific key-value pair in the user's long-term memory."""
    try:
        user_id: str = request.state.user_id
        logger.info("Upserting LTM key='%s' for user=%s", body.key, user_id)
        saved_key = await upsert_long_term_memory(user_id=user_id, key=body.key, value=body.value)
        return JSONResponse(content={
            "success": True,
            "message": f"Memory '{saved_key}' saved successfully",
            "data": {"key": saved_key, "value": body.value},
        })
    except Exception as e:
        logger.error("Upsert LTM failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.delete("/long_term_memory/{key}")
async def delete_long_term_memory_key_endpoint(request: Request, key: str):
    """Delete a single key from the user's long-term memory."""
    try:
        user_id: str = request.state.user_id
        logger.info("Deleting LTM key='%s' for user=%s", key, user_id)
        await delete_long_term_memory_key(user_id=user_id, key=key)
        return JSONResponse(content={
            "success": True,
            "message": f"Memory key '{key}' deleted successfully",
            "data": None,
        })
    except Exception as e:
        logger.error("Delete LTM key failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.delete("/long_term_memory")
async def delete_all_long_term_memory(request: Request):
    """Delete ALL long-term memory entries for this user."""
    try:
        user_id: str = request.state.user_id
        logger.info("Deleting ALL LTM for user=%s", user_id)
        memories = await get_user_long_term_memory(user_id=user_id)
        for item in memories:
            key = item.get("key")
            if key:
                await delete_long_term_memory_key(user_id=user_id, key=key)
        return JSONResponse(content={
            "success": True,
            "message": f"All {len(memories)} memory entries deleted",
            "data": None,
        })
    except Exception as e:
        logger.error("Delete all LTM failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})
