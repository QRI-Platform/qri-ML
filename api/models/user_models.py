from pydantic import BaseModel


class GetUserThreads(BaseModel):
    user_id:str



class GetUserConversation(BaseModel):
    user_id:str
    thread_id:str