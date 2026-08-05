from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Optional, Any
from operator import add


class OrchastratorOutput(BaseModel):
    require_db_search: bool = Field(
        description="Set to true if the user query requires searching the vector database, false if it can be answered directly."
    )

    @field_validator("require_db_search", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v


class QueryGenerationOutput(BaseModel):
    queries: List[str]


class ChatOutput(BaseModel):
    response: str = Field(description="Answer to the user")
    memory_key: Optional[str] = Field(default=None, description="Key of user memory item to save or update if user provided key details or preferences, else null")
    memory_value: Optional[str] = Field(default=None, description="Value of user memory item to save or update, else null")


class State(BaseModel):
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)
    require_db_search: bool = False
    has_documents: bool = False  # True when Pinecone namespace has vectors for this thread
    queries: List[str] = Field(default_factory=list)
    retreived_results: List[Any] = Field(default_factory=list)
    ai_response: Optional[str] = None


Orchastrator_output = OrchastratorOutput
Query_generation_output = QueryGenerationOutput
