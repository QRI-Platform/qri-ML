from fastapi import Request, HTTPException
from src.logger import logger


async def authenticate_user(request: Request, user_id: str, thread_id: str):
    try:
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
