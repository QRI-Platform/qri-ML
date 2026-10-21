
from src.domain.state import State,PerformanceGenerationSchema
from src.prompts.templates import PERFORMANCE_PROMPT
from src.llm.llm_loader import get_llm
from src.core.constants import DEFAULT_TOTAL_NO_OF_QUESTIONS
from src.core.exceptions import MyException
from src.core.logger import logger
import sys


async def performance_node(state:State):
    """This node generates the performance for the solved_test_paper"""
    try:
        llm = get_llm(reasoning_format=None, reasoning_effort=None)
        structured_llm = llm.with_structured_output(
            PerformanceGenerationSchema,
            method="json_schema",
            strict=True,
        )

        chain = PERFORMANCE_PROMPT | structured_llm

        result = await chain.ainvoke({"solved_test_paper": state.get("solved_test_paper", {})})
        results = result.model_dump()

        results["total_correct_answers"] = state.get("total_correct_answers", 0)
        results["total_questions"] = state.get("total_questions", DEFAULT_TOTAL_NO_OF_QUESTIONS)
        return {"performance_schema": results, "need_performance_generation": False}

    except Exception as e:
        logger.error("performance_node failed: %s", str(e))
        raise MyException(e, sys)
