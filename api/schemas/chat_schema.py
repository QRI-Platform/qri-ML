from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    metadata:dict = Field({}, description="Optional metadata to pass to the chat pipeline")


class TestGenerationRequest(BaseModel):
    total_no_of_questions: int = Field(2, description="Number of questions to generate")
    level: Literal['easy', 'medium', 'hard'] = Field('medium', description="Difficulty level of the questions")
    subject_name: str = Field("maths", description="Subject for which questions should be generated")
    exam_type: str = Field("IIT_JEE", description="Exam type the questions should match")
                           