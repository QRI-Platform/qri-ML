from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage
from typing import Annotated, List, Optional, Any, TypedDict
from langgraph.graph.message import add_messages


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


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    file_paths: List[str]
    require_db_search: bool
    has_documents: bool  # True when Pinecone namespace has vectors for this thread
    queries: List[str]
    retreived_results: List[Any]
    ai_response: Optional[str]


Orchastrator_output = OrchastratorOutput
Query_generation_output = QueryGenerationOutput

