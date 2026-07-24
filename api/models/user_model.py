from pydantic import BaseModel,Field
from typing import Optional


class SetUserThreadTimeLimit(BaseModel):
    thread_id:Optional[str] = Field(default=None,description="this is the unique id from frontend to set for a particular chat session")
    time_duration:int = Field(1*60,description="time the users uploaded content will persist")