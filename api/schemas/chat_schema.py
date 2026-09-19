from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    metadata:dict = Field({}, description="Optional metadata to pass to the chat pipeline")
