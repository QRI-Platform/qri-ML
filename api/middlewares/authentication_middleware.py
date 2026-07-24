from fastapi import Request, HTTPException
from src.logger import logger

async def authenticate_user(request:Request,thread_id: str):
    try:
        if not thread_id:
            raise HTTPException(
                status_code=401,
                detail={"success": False, "message": "Please login", "data": None}
            )
        request.state.thread_id = thread_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=401,
            detail={"success": False, "message": "Please login", "data": None}
        )
