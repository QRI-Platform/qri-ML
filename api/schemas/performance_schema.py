from pydantic import BaseModel, Field
from typing import Literal
from src.core.constants import DEFAULT_EXAM_TYPE,DEFAULT_SUBJECT_NAME,DEFAULT_TOTAL_NO_OF_QUESTIONS

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    metadata:dict = Field({}, description="Optional metadata to pass to the chat pipeline")


class PerformanceGenerationRequest(BaseModel):
    total_correct_answers: int = Field(0, description="Number of correct answeres")
    total_questions: int= Field(DEFAULT_TOTAL_NO_OF_QUESTIONS, description="Total no of questions")
    solved_test_paper: dict = Field({}, description="Solved test paper")
                           