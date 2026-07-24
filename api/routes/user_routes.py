from src.logger import logger
from fastapi import APIRouter,Request,BackgroundTasks
from fastapi.responses import JSONResponse
from api.helper.graph_helper import delete_thread
from api.models.user_model import SetUserThreadTimeLimit
# from src.constants import COOKIE_MAX_AGE_SECONDS
import uuid
router = APIRouter()




# ===================== User Login for a perticular fixed session ===============================

@router.get("/login/{time_duration}")
async def login_user(set_user_thread_time_limit:SetUserThreadTimeLimit,background_tasks: BackgroundTasks):
    thread_id=set_user_thread_time_limit.thread_id
    time_duration = set_user_thread_time_limit.time_duration
    logger.info(f"login_user called for thread_id: {thread_id}")
    background_tasks.add_task(delete_thread, thread_id=thread_id, delay_seconds=time_duration*60)
    logger.info(f"deletion of thread id has been seted up {time_duration} minutes")
    response = JSONResponse(content={"success": True, "message": "User logged in successfully", "data": thread_id})

    return response


