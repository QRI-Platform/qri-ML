from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    metadata:dict = Field({}, description="Optional metadata to pass to the chat pipeline")


class TestGenerationRequest(BaseModel):
    total_no_of_questions:int=Field(10,description=""),
    no_of_easy_questions:int=Field(2,description=""),
    no_of_medium_questions:int=Field(4,description=""),
    no_of_hard_questions:int=Field(4,description=""),
    level:Literal['easy','medium','hard']=Field('medium',description=""),
    subject_name:str=Field("maths",description=""),
    exam_type:str=Field("IIT_JEE",description="")
                           