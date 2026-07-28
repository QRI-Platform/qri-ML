from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from src.logger import logger
from db.thread_manager import get_thread_manager,ThreadManager
from src.memory import get_checkpointer
from api.helper.graph_helper import load_conversation
from api.models.user_models import (
    GetUserThreads,
    GetUserConversation
)
import json
router:APIRouter = APIRouter()


@router.get("/threads")
async def get_user_all_threads(payload:GetUserThreads):
    try:
        user_id:str = payload.user_id
        
        logger.info("retreiving user all threads %s",user_id)
        thread_manager:ThreadManager = get_thread_manager()

        user_threads=thread_manager.get_threads(user_id=user_id)

        return JSONResponse(content={"data":user_threads,"message":"retreived user threads","success":True},status_code=200)

    except Exception as e:
        logger.error("err while retreiving user threads")
        raise HTTPException(status_code=400,detail={"data":None,"messages":str(e),"success":False})


@router.get("/conversation")
async def get_user_conversation(payload:GetUserConversation):
    try:
        user_id:str = payload.user_id
        thread_id:str = payload.thread_id
        logger.info("Retreiving user Conversation")
        messages=await load_conversation(thread_id=thread_id)

        return JSONResponse(content={"data":messages,"message":"retreived data succesfully","success":True})

    except Exception as e:
        logger.error("err while loading user conversations")
        raise HTTPException(status_code=400,detail={"data":None,"messages":str(e),"success":False})


        


