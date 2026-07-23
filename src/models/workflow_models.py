from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage
from typing import Annotated, List, Optional, Any
from operator import add


class Orchastrator_output(BaseModel):
    require_db_search: bool = Field(description="Set to true if the user query requires searching the vector database, false if it can be answered directly.")

    @field_validator("require_db_search", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v


class Query_generation_output(BaseModel):
    queries: List[str]


class State(BaseModel):
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    user_id: str
    file_paths: List[str] = Field(default_factory=list)
    retreiver: Optional[Any] = None
    require_db_search: bool = False
    queries: List[str] = Field(default_factory=list)
    retreived_results: List[Any] = Field(default_factory=list)
    ai_response: Optional[str] = None