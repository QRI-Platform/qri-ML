from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chat pipeline")
    has_documents: bool = Field(..., description="Indicates if the user has documents to process")
