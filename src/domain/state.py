from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage
from typing import Annotated, List, Optional, Any, TypedDict,Literal
from langgraph.graph.message import add_messages
from src.core.constants import DEFAULT_DEFFICULTY_LEVEL,DEFAULT_EXAM_TYPE,DEFAULT_SUBJECT_NAME,DEFAULT_TOTAL_NO_OF_QUESTIONS



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
    



# ================= Test-Generation ===============================
class Options(BaseModel):
    option1: str = Field(..., description="First choice text")
    option2: str = Field(..., description="Second choice text")
    option3: str = Field(..., description="Third choice text")
    option4: str = Field(..., description="Fourth choice text")

class Question(BaseModel):
    question_text: str = Field(..., description="The main challenge or prompt for the user")
    options: Options = Field(..., description="The pool of 4 choices")
    correct: Literal['option1', 'option2', 'option3', 'option4'] = Field(
        ..., description="The key of the correct option"
    )
    
    explanation: str = Field(
        ..., description="Educational feedback displayed to the user after they answer"
    )
    hint: Optional[str] = Field(
        None, description="An optional hint to help the player if they struggle"
    )
    difficulty: Literal['easy', 'medium', 'hard'] = Field(
        'medium', description="Difficulty tier used for scoring adjustments"
    )
    points: int = Field(
        10, description="Base XP or score awarded for answering this question correctly"
    )
    

class Questions_generation_schema(BaseModel):
    questions: List[Question] = Field(..., description="A sequence of game-ready questions")




class PerformanceGenerationSchema(BaseModel):
    weak_area: list[str] = Field(
        default_factory=list,
        description="Topics where the user demonstrates weak understanding based on incorrect answers and recurring mistakes."
    )

    strong_area: list[str] = Field(
        default_factory=list,
        description="Topics where the user demonstrates consistent accuracy and strong conceptual understanding."
    )

    mid_area: list[str] = Field(
        default_factory=list,
        description="Topics where the user demonstrates partial understanding or inconsistent performance."
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Specific actions or practice suggestions to improve the user's weak and intermediate areas."
    )

    overall_feedback: str = Field(
        description="A concise summary of the user's overall performance based on the analyzed answers."
    )




class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    file_paths: List[str]
    require_db_search: bool
    has_documents: bool = False  # True when Pinecone namespace has vectors for this thread
    queries: List[str]
    retreived_results: List[Any]
    ai_response: Optional[str]
    need_title:bool = False


    # Test Generation logic
    need_test_paper:bool=False
    test_paper:Any = None
    total_no_of_questions:int=DEFAULT_TOTAL_NO_OF_QUESTIONS
    level:str=DEFAULT_DEFFICULTY_LEVEL
    subject_name:str=DEFAULT_SUBJECT_NAME
    exam_type:str=DEFAULT_EXAM_TYPE



    # Test Performance logic
    need_performance_generation:bool=False
    total_correct_answers:int=0
    total_questions:int=DEFAULT_TOTAL_NO_OF_QUESTIONS
    solved_test_paper:dict={}
    performance_schema:dict={}






Orchestrator_output = OrchestratorOutput
OrchastratorOutput = OrchestratorOutput
Orchastrator_output = OrchestratorOutput
Query_generation_output = QueryGenerationOutput




