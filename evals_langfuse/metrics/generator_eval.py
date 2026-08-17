import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore
from evals.common import get_llm, EvaluationResult
from langchain_core.prompts import ChatPromptTemplate
from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.nodes.main_nodes import chat_node


# --- Pydantic Schema for LLM Judge ---
class AccuracyOutput(BaseModel):
    score: float = Field(
        ..., 
        description="Float score between 0.0 and 1.0 indicating how closely content and intent match the expected response."
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation justifying the assigned score."
    )


async def eval_generator_target(inputs: dict) -> dict:
    """Target function under evaluation wrapping chat_node.

    Args:
        inputs: Dict containing 'question', optional 'messages', 
                optional 'retreived_results', and optional mock memory store.
    """
    try:
        logger.info("Executing eval_generator_target")

        # 1. Prepare conversation messages
        messages = inputs.get("messages")
        if not messages:
            question = inputs.get("question", "Hello, this is a test message.")
            messages = [HumanMessage(content=question)]

        # 2. Prepare retrieved documents (if any)
        raw_docs = inputs.get("retreived_results", [])
        retreived_docs = []
        for doc in raw_docs:
            if isinstance(doc, str):
                retreived_docs.append(Document(page_content=doc))
            elif isinstance(doc, dict):
                retreived_docs.append(Document(page_content=doc.get("page_content", "")))
            else:
                retreived_docs.append(doc)

        # 3. Construct state
        state: State = {
            "messages": messages,
            "retreived_results": retreived_docs,
        }

        user_id = inputs.get("user_id", "test_user_id")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "eval_thread_id",
                "user_id": user_id,
            }
        }

        # 4. Setup in-memory store & pre-populate existing memories
        store = InMemoryStore()
        initial_memories = inputs.get("user_memories", {})
        for k, v in initial_memories.items():
            await store.aput(namespace=("user", str(user_id), "details"), key=k, value={"data": v})

        # 5. Execute chat_node
        res = await chat_node(state, config, store)

        # 6. Extract predictions
        last_msg = res.get("messages", [])[-1] if res.get("messages") else None
        tool_calls = getattr(last_msg, "tool_calls", []) if last_msg else []
        has_tool_call = len(tool_calls) > 0
        ai_response = res.get("ai_response", getattr(last_msg, "content", ""))

        return {
            "ai_response": ai_response,
            "has_tool_call": has_tool_call,
            "tool_calls": tool_calls,
            "messages": res.get("messages", []),
        }

    except Exception as e:
        logger.error("Error in eval_generator_target: %s", str(e), exc_info=True)
        raise MyException(e, sys)


async def eval_generator(input: dict = None, output: dict = None, expected_output: dict = None, **kwargs) -> EvaluationResult:
    """Evaluator function comparing predicted output against ground truth.
    
    Evaluates:
      - Tool call trigger correctness (if expected)
      - Semantic content alignment via LLM Judge
    """
    try:
        inp = input or kwargs.get("inputs", {})
        out = output or kwargs.get("outputs", {})
        exp = expected_output or kwargs.get("reference_outputs", {})

        expected_tool_call = exp.get("has_tool_call", False)
        actual_tool_call = out.get("has_tool_call", False)

        # Tool Call Accuracy Check
        if expected_tool_call != actual_tool_call:
            return EvaluationResult(
                name="generator_evaluation",
                value=0.0,
                comment=f"Tool call mismatch. Expected: {expected_tool_call}, Got: {actual_tool_call}",
                metadata={
                    "tool_call_matched": False,
                },
            )

        # If a tool call was expected and triggered properly, give full score
        if expected_tool_call and actual_tool_call:
            return EvaluationResult(
                name="generator_evaluation",
                value=1.0,
                comment="Correctly routed to tool node with tool_calls.",
                metadata={
                    "tool_call_matched": True,
                },
            )

        # Semantic Content Evaluation for Direct AI Responses
        expected_response = exp.get("ai_response", "")
        actual_response = out.get("ai_response", "")

        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            """You are an accurate AI evaluation judge.
Compare the generated LLM response with the expected reference response.

[Context Question]: {question}
[Expected Response]: {expected_response}
[Generated LLM Response]: {llm_response}

Score the similarity and quality between 0.0 and 1.0."""
        )

        chain = prompt | llm.with_structured_output(AccuracyOutput)
        judge_result = await chain.ainvoke({
            "question": inp.get("question", ""),
            "expected_response": expected_response,
            "llm_response": actual_response,
        })

        return EvaluationResult(
            name="generator_accuracy",
            value=judge_result.score,
            comment=judge_result.reasoning,
            metadata={
                "tool_call_matched": True,
                "reasoning": judge_result.reasoning,
            },
        )

    except Exception as e:
        logger.error("Error in eval_generator evaluator: %s", str(e), exc_info=True)
        return EvaluationResult(
            name="generator_accuracy",
            value=0.0,
            comment=f"Evaluator exception: {str(e)}",
        )
     



