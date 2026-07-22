from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from typing import Annotated
from operator import add

# main state of the graph flow
class State(BaseModel):
    messages:Annotated[BaseMessage,add]
