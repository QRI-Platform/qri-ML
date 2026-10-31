from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.tools.retreiver_tool import retreiver
from src.tools.save_long_term_memory_tool import save_long_term_memory
from src.tools.solver_tool import solver


class TestSolverToolUnit:

    @pytest.mark.asyncio
    async def test_solver_basic_arithmetic(self):
        result = await solver.ainvoke({"expression": "2 + 2"})
        assert result == "4"

    @pytest.mark.asyncio
    async def test_solver_complex_expression(self):
        result = await solver.ainvoke({"expression": "sqrt(144) + 5**2"})
        assert float(result) == 37.0

    @pytest.mark.asyncio
    async def test_solver_math_constants(self):
        result = await solver.ainvoke({"expression": "sin(pi / 2)"})
        assert float(result) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_solver_strips_backticks_and_quotes(self):
        result = await solver.ainvoke({"expression": "`10 * 5`"})
        assert result == "50"

    @pytest.mark.asyncio
    async def test_solver_invalid_expression_returns_error_message(self):
        result = await solver.ainvoke({"expression": "invalid + expression *"})
        assert "Error: Unable to evaluate mathematical expression" in result

    @pytest.mark.asyncio
    async def test_solver_blocks_unauthorized_imports_or_code(self):
        result = await solver.ainvoke({"expression": "__import__('os').system('ls')"})
        assert "Error: Unable to evaluate mathematical expression" in result


class TestSaveLongTermMemoryToolUnit:

    @pytest.mark.asyncio
    async def test_save_valid_memory_success(self, mock_base_store):
        config = {
            "configurable": {
                "user_id": "user_42",
                "metadata": {"plan": {"long_term_memory_cap": 5}},
            }
        }
        mock_base_store.asearch.return_value = []

        result = await save_long_term_memory.coroutine(
            memory_key="Favorite Color",
            memory_value="Deep Blue",
            config=config,
            store=mock_base_store,
        )

        assert "Successfully saved memory: favorite_color = Deep Blue" in result
        mock_base_store.aput.assert_called_once_with(
            namespace=("user", "user_42", "details"),
            key="favorite_color",
            value={"data": "Deep Blue"},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_key", ["", "   ", "none", "null", "undefined"])
    async def test_save_invalid_memory_key_rejected(self, mock_base_store, invalid_key):
        config = {
            "configurable": {
                "user_id": "user_42",
            }
        }

        result = await save_long_term_memory.coroutine(
            memory_key=invalid_key,
            memory_value="Some value",
            config=config,
            store=mock_base_store,
        )

        assert "Failed: Invalid memory key provided" in result
        mock_base_store.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_memory_exceeds_cap_rejected(self, mock_base_store):
        config = {
            "configurable": {
                "user_id": "user_42",
                "metadata": {"plan": {"long_term_memory_cap": 2}},
            }
        }
        # Simulate 2 already stored memories
        item1 = MagicMock()
        item1.key = "key1"
        item2 = MagicMock()
        item2.key = "key2"
        mock_base_store.asearch.return_value = [item1, item2]

        result = await save_long_term_memory.coroutine(
            memory_key="new_key",
            memory_value="new_value",
            config=config,
            store=mock_base_store,
        )

        assert "Failed: Long-term memory cap of 2 reached" in result
        mock_base_store.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_memory_updates_existing_key_when_at_cap(self, mock_base_store):
        config = {
            "configurable": {
                "user_id": "user_42",
                "metadata": {"plan": {"long_term_memory_cap": 2}},
            }
        }
        mock_base_store.asearch.return_value = ["key1", "key2"]

        result = await save_long_term_memory.coroutine(
            memory_key="key1",
            memory_value="updated_value",
            config=config,
            store=mock_base_store,
        )
        assert "Successfully saved memory" in result


class TestRetrieverToolUnit:

    @pytest.mark.asyncio
    @patch("src.tools.retreiver_tool.get_retriever")
    async def test_retreiver_tool_parallel_queries(self, mock_get_retriever):
        mock_retriever = AsyncMock()
        mock_vs = MagicMock()
        mock_retriever.create_retriever.return_value = mock_vs

        doc1 = Document(page_content="doc 1 content")
        doc2 = Document(page_content="doc 2 content")
        mock_retriever.get_similar_documents.side_effect = [[doc1], [doc2]]
        mock_get_retriever.return_value = mock_retriever

        config = {
            "configurable": {
                "thread_id": "th-999",
            }
        }

        output = await retreiver.ainvoke(
            {"queries": ["query 1", "query 2"], "filters": ["guide.pdf"]},
            config=config,
        )

        assert "retreived_results" in output
        assert len(output["retreived_results"]) == 2
        # Verify prefix added to filter filenames: th-999_guide.pdf
        mock_retriever.get_similar_documents.assert_any_call(
            vector_store=mock_vs, query="query 1", filter=["th-999_guide.pdf"]
        )
