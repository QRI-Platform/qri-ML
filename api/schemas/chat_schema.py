from pydantic import BaseModel, Field
from typing import Literal
from src.core.constants import DEFAULT_DEFFICULTY_LEVEL,DEFAULT_EXAM_TYPE,DEFAULT_SUBJECT_NAME,DEFAULT_TOTAL_NO_OF_QUESTIONS

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    metadata:dict = Field({}, description="Optional metadata to pass to the chat pipeline")


class TestGenerationRequest(BaseModel):
    total_no_of_questions: int = Field(DEFAULT_TOTAL_NO_OF_QUESTIONS, description="Number of questions to generate")
    level: str= Field(DEFAULT_DEFFICULTY_LEVEL, description="Difficulty level of the questions")
    subject_name: str = Field(DEFAULT_SUBJECT_NAME, description="Subject for which questions should be generated")
    exam_type: str = Field(DEFAULT_EXAM_TYPE, description="Exam type the questions should match")
                           