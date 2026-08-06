from fastapi import Request, HTTPException
from src.core.logger import logger


# takes user_id and thread_id as input in the params or headers
async def authenticate_user(request: Request, user_id: str | None = None, thread_id: str | None = None):
    try:
        user_id = user_id or request.query_params.get("user_id") or request.headers.get("user_id") or request.headers.get("x-user-id")
        thread_id = thread_id or request.query_params.get("thread_id") or request.headers.get("thread_id") or request.headers.get("x-thread-id")

        if not user_id or not thread_id:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "user_id and thread_id are required", "data": None}
            )
        request.state.user_id = user_id
        request.state.thread_id = thread_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication error: %s", e)
        raise HTTPException(
            status_code=401,
            detail={"success": False, "message": "Authentication failed", "data": None}
        )
