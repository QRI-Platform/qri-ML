from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage
from typing import Annotated, List, Optional, Any, TypedDict
from langgraph.graph.message import add_messages


class OrchestratorOutput(BaseModel):
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
    queries: List[str] = Field(description="write the user refined optimised queries for vector db search")


class ChatOutput(BaseModel):
    response: str = Field(description="Answer to the user strictly in markdown formate including emojis if needed")
    

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    file_paths: List[str]
    require_db_search: bool
    has_documents: bool = False  # True when Pinecone namespace has vectors for this thread
    queries: List[str]
    retreived_results: List[Any]
    ai_response: Optional[str]
    need_title:bool = False


Orchestrator_output = OrchestratorOutput
OrchastratorOutput = OrchestratorOutput
Orchastrator_output = OrchestratorOutput
Query_generation_output = QueryGenerationOutput
