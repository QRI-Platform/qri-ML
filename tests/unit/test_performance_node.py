from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.constants import DEFAULT_TOTAL_NO_OF_QUESTIONS
from src.core.exceptions import MyException
from src.domain.state import PerformanceGenerationSchema
from src.nodes.performance_node import performance_node


class TestPerformanceNodeUnit:

    @pytest.mark.asyncio
    @patch("src.nodes.performance_node.get_llm")
    async def test_performance_node_success(self, mock_get_llm):
        mock_schema_output = PerformanceGenerationSchema(
            weak_area=["Trigonometry", "Integration"],
            strong_area=["Algebra", "Linear Equations"],
            mid_area=["Probability"],
            recommendations=["Focus on practice problems for trigonometric identities"],
            overall_feedback="Solid grasp on foundational algebra; needs work on calculus.",
        )

        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_schema_output)
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        state = {
            "solved_test_paper": {
                "questions": [
                    {
                        "question": "What is sin(90)?",
                        "user_answer": "0",
                        "correct_answer": "1",
                        "topic": "Trigonometry",
                    },
                    {
                        "question": "Solve 2x = 4",
                        "user_answer": "2",
                        "correct_answer": "2",
                        "topic": "Algebra",
                    },
                ]
            },
            "total_correct_answers": 1,
            "total_questions": 2,
        }

        # Mock the RunnableBinding/chain pipe: PERFORMANCE_PROMPT | structured_llm
        with patch("src.nodes.performance_node.PERFORMANCE_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_schema_output)
            mock_prompt.__or__.return_value = mock_chain

            result = await performance_node(state)

        assert "performance_schema" in result
        assert result["need_performance_generation"] is False

        schema = result["performance_schema"]
        assert schema["weak_area"] == ["Trigonometry", "Integration"]
        assert schema["strong_area"] == ["Algebra", "Linear Equations"]
        assert schema["mid_area"] == ["Probability"]
        assert schema["recommendations"] == ["Focus on practice problems for trigonometric identities"]
        assert schema["overall_feedback"] == "Solid grasp on foundational algebra; needs work on calculus."
        assert schema["total_correct_answers"] == 1
        assert schema["total_questions"] == 2

    @pytest.mark.asyncio
    @patch("src.nodes.performance_node.get_llm")
    async def test_performance_node_default_fallback_values(self, mock_get_llm):
        mock_schema_output = PerformanceGenerationSchema(
            weak_area=[],
            strong_area=[],
            mid_area=[],
            recommendations=[],
            overall_feedback="Inconclusive test paper.",
        )

        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        # State without total_correct_answers and total_questions
        state = {
            "solved_test_paper": {},
        }

        with patch("src.nodes.performance_node.PERFORMANCE_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=mock_schema_output)
            mock_prompt.__or__.return_value = mock_chain

            result = await performance_node(state)

        schema = result["performance_schema"]
        assert schema["total_correct_answers"] == 0
        assert schema["total_questions"] == DEFAULT_TOTAL_NO_OF_QUESTIONS
        assert result["need_performance_generation"] is False

    @pytest.mark.asyncio
    @patch("src.nodes.performance_node.get_llm")
    async def test_performance_node_failure_raises_my_exception(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        state = {"solved_test_paper": {}}

        with patch("src.nodes.performance_node.PERFORMANCE_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM API call failed"))
            mock_prompt.__or__.return_value = mock_chain

            with pytest.raises(MyException):
                await performance_node(state)
