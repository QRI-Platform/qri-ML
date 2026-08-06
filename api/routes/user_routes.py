from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field
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


@router.get(
    "/conversation",
    summary="Get Conversation History",
    responses={
        200: {"description": "List of serialized LangChain messages (HumanMessage, AIMessage, ToolMessage etc.) with all fields including tool_calls and reasoning."},
        400: {"description": "Failed to load conversation from checkpointer."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def get_user_conversation(request: Request):
    """
    Retrieve the full conversation history for the given `thread_id`.

    Returns a list of serialized LangChain messages. Each message object includes:
    - `type` — `"human"`, `"ai"`, `"tool"`
    - `content` — the message text
    - `tool_calls` — tool invocations made by the AI (if any)
    - `additional_kwargs` — reasoning / thinking content (if model supports it)
    - `usage_metadata` — token counts
    - `response_metadata` — model name, finish reason etc.
    """
    try:
        user_id: str = request.state.user_id
        thread_id: str = request.state.thread_id
        logger.info("Retrieving user conversation")
        messages = await load_conversation(thread_id=thread_id, user_id=user_id)
        return JSONResponse(content={"data": messages, "message": "retrieved data successfully", "success": True})
    except Exception as e:
        logger.error("Error while loading user conversations: %s", str(e))
        raise HTTPException(status_code=400, detail={"data": None, "messages": str(e), "success": False})


@router.get(
    "/long_term_memory/",
    summary="Get User Long-Term Memory",
    responses={
        200: {"description": "List of key-value memory items stored for this user across all sessions."},
        400: {"description": "Failed to retrieve memory from BaseStore."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def get_users_long_term_memory(request: Request):
    """
    Retrieve all long-term memory entries for the user.

    Long-term memory is automatically extracted by the LLM during conversations
    (e.g. user preferences, personal details) and stored in the LangGraph BaseStore.
    These memories are injected into every future conversation for this user,
    regardless of `thread_id`.

    Each entry has:
    - `key` — the memory key (e.g. `"preferred_language"`)
    - `value` — the stored dict (e.g. `{"data": "Python"}`)
    """
    try:
        logger.info("Retrieving user long term memory for %s", request.state.user_id)
        messages = await get_user_long_term_memory(user_id=request.state.user_id)
        return JSONResponse(content={"data": messages, "message": "retrieved long term memory successfully", "success": True})
    except Exception as e:
        logger.error("Error while retrieving long term memory: %s", str(e))
        raise HTTPException(status_code=400, detail={"data": None, "messages": str(e), "success": False})


@router.delete(
    "/pine_cone",
    summary="Delete Pinecone Namespace (Vector Data)",
    responses={
        200: {"description": "Pinecone namespace for this thread deleted successfully."},
        400: {"description": "Pinecone deletion failed."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def delete_thread_endpoint(request: Request):
    """
    Delete all vectors stored in the Pinecone namespace for the given `thread_id`.

    This removes all document chunks that were ingested for this thread.
    The conversation history in the LangGraph checkpointer is **not** affected —
    use `DELETE /conversation` for that.
    """
    try:
        logger.info("delete endpoint: user=%s thread=%s", request.state.user_id, request.state.thread_id)
        await delete_pinecone_namespace(request.state.thread_id)
        logger.info("delete endpoint: completed for thread=%s", request.state.thread_id)
        return JSONResponse(content={"success": True, "message": "Thread deleted successfully", "data": None}, status_code=200)
    except Exception as e:
        logger.error("delete endpoint failed: %s", str(e))
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


@router.delete(
    "/conversation",
    summary="Delete Conversation History",
    responses={
        200: {"description": "Conversation thread deleted from LangGraph checkpointer."},
        400: {"description": "Deletion failed."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def delete_user_conversation_endpoint(request: Request):
    """
    Delete the conversation history (LangGraph checkpoint) for the given `thread_id`.

    This removes all message history from the checkpointer (SQLite).
    Pinecone vector data is **not** affected — use `DELETE /pine_cone` for that.
    """
    try:
        success = await delete_user_conversation(thread_id=request.state.thread_id, user_id=request.state.user_id)
        if success:
            return JSONResponse(content={"success": success, "message": "Conversation deleted", "data": None})
        else:
            return JSONResponse(content={"success": success, "message": "Conversation not found", "data": None})
    except Exception as e:
        raise HTTPException(status_code=400, detail={"success": False, "message": str(e), "data": None})


# ── Long-Term Memory Management ──────────────────────────────────────────────

class UpsertMemoryRequest(BaseModel):
    key: str = Field(..., description="Memory key (e.g. 'preferred_language'). Stored lowercase with underscores.")
    value: str = Field(..., description="Memory value to store (e.g. 'Python').")


@router.post(
    "/long_term_memory",
    summary="Add or Update a Long-Term Memory Entry",
    responses={
        200: {"description": "Memory key upserted successfully."},
        400: {"description": "Failed to upsert memory."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def add_or_update_long_term_memory(request: Request, body: UpsertMemoryRequest):
    """
    Manually add or update a key-value pair in the user's long-term memory.

    Keys are normalized to `lowercase_with_underscores`. If the key already exists,
    the value is overwritten. These memories are injected into all future chat
    sessions for this user automatically.
    """
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


@router.delete(
    "/long_term_memory/{key}",
    summary="Delete a Single Long-Term Memory Key",
    responses={
        200: {"description": "Memory key deleted successfully."},
        400: {"description": "Deletion failed."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def delete_long_term_memory_key_endpoint(request: Request, key: str):
    """
    Delete a single key from the user's long-term memory by its key name.

    The `key` path parameter must match the normalized form (lowercase, underscores).
    Use `GET /long_term_memory/` to see all existing keys.
    """
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


@router.delete(
    "/long_term_memory",
    summary="Delete ALL Long-Term Memory Entries",
    responses={
        200: {"description": "All memory entries for this user deleted."},
        400: {"description": "Deletion failed."},
        401: {"description": "Missing or invalid user_id / thread_id."},
    },
)
async def delete_all_long_term_memory(request: Request):
    """
    Delete **all** long-term memory entries for the user.

    This is a bulk delete — all keys in the user's memory namespace are removed.
    The user will start fresh with no persisted memories in future sessions.
    """
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
