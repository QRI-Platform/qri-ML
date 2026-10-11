import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from src.nodes.main_nodes import (
    ingestion_node,
    orchastrator_node,
    query_generation_node,
    retreiver_node,
    test_generation_node as _test_generation_node,
)
_test_generation_node.__test__ = False
from src.nodes.advance_nodes import summerizer, title_renamer_node
from src.domain.state import OrchestratorOutput, QueryGenerationOutput, Questions_generation_schema, Question, Options
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP, MAX_TOTAL_NO_OF_QUESTIONS_PER_REQUEST


class TestNodesUnit:

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.DataIngestion")
    @patch("src.nodes.main_nodes.get_retriever")
    async def test_ingestion_node_success(self, mock_get_retriever, mock_data_ingestion_cls):
        mock_ingestion_instance = MagicMock()
        mock_ingestion_instance.ingest = AsyncMock()
        mock_data_ingestion_cls.return_value = mock_ingestion_instance

        state = {"file_paths": ["/tmp/upload/guide.pdf", "/tmp/upload/sheet.xlsx"]}
        config = {"configurable": {"thread_id": "thread-123"}}

        result = await ingestion_node(state, config)

        mock_ingestion_instance.ingest.assert_called_once()
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert "guide.pdf" in result["messages"][0].content
        assert "sheet.xlsx" in result["messages"][1].content

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.get_llm")
    async def test_orchastrator_node_db_search_true_when_documents_exist(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = OrchestratorOutput(require_db_search=True)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="Summarize the uploaded document")],
            "has_documents": True,
        }
        config = {"configurable": {"thread_id": "th-1"}}

        result = await orchastrator_node(state, config)
        assert result["require_db_search"] is True
        assert result["has_documents"] is True

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.get_llm")
    async def test_orchastrator_node_safety_override_when_no_documents(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        # LLM erroneously requests DB search even though has_documents is False
        mock_structured.ainvoke.return_value = OrchestratorOutput(require_db_search=True)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="Search my notes")],
            "has_documents": False,
        }
        config = {"configurable": {"thread_id": "th-1"}}

        result = await orchastrator_node(state, config)
        # Should be overridden to False by safety guard
        assert result["require_db_search"] is False
        assert result["has_documents"] is False

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.get_llm")
    async def test_query_generation_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = QueryGenerationOutput(
            queries=["query 1: attention mechanism", "query 2: transformer architecture"]
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="How does attention work?")]}
        result = await query_generation_node(state)

        assert "queries" in result
        assert len(result["queries"]) == 2
        assert "query 1: attention mechanism" in result["queries"]

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.get_retriever")
    async def test_retreiver_node_with_at_mention_filtering(self, mock_get_retriever):
        mock_retriever = AsyncMock()
        mock_vs = MagicMock()
        mock_retriever.create_retriever.return_value = mock_vs
        mock_retriever.get_similar_documents.return_value = [
            MagicMock(page_content="relevant chunk")
        ]
        mock_get_retriever.return_value = mock_retriever

        state = {
            "queries": ["deep learning concept"],
            "messages": [HumanMessage(content="Check details in @paper.pdf")],
        }
        config = {"configurable": {"thread_id": "th-abc"}}

        result = await retreiver_node(state, config)
        assert "retreived_results" in result

        # Verify filter applied reconstructed name: th-abc_paper.pdf
        mock_retriever.get_similar_documents.assert_called_once_with(
            vector_store=mock_vs,
            query="deep learning concept",
            filter=["th-abc_paper.pdf"],
        )

    @pytest.mark.asyncio
    @patch("src.nodes.advance_nodes.get_llm")
    async def test_title_renamer_node(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Machine Learning Basics"
        mock_llm.invoke.return_value = mock_resp
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="Can you explain neural networks to me?")]
        }
        result = await title_renamer_node(state)

        assert result["title"] == "Machine Learning Basics"
        assert result["need_title"] is False

    @pytest.mark.asyncio
    async def test_summerizer_skips_when_few_messages(self):
        # When messages count <= NO_OF_LAST_MESSAGES_TO_KEEP
        msgs = [HumanMessage(content="Hi", id="1"), AIMessage(content="Hello", id="2")]
        state = {"messages": msgs}
        config = {"configurable": {"thread_id": "th-1"}}

        result = await summerizer(state, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_summerizer_skips_when_messages_have_no_ids(self):
        # Messages count exceeds threshold but no id attributes
        msgs = [HumanMessage(content=f"msg {i}", id=None) for i in range(NO_OF_LAST_MESSAGES_TO_KEEP + 3)]
        state = {"messages": msgs}
        config = {"configurable": {"thread_id": "th-1"}}

        result = await summerizer(state, config)
        assert result == {}

    @pytest.mark.asyncio
    @patch("src.nodes.advance_nodes.get_llm")
    async def test_summerizer_summarizes_and_deletes_old_messages(self, mock_get_llm):
        mock_llm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "User discussed Python and Docker."
        mock_llm.ainvoke.return_value = mock_resp
        mock_get_llm.return_value = mock_llm

        # Create enough messages with IDs
        msgs = [
            HumanMessage(content=f"old msg {i}", id=f"id_{i}")
            for i in range(NO_OF_LAST_MESSAGES_TO_KEEP + 3)
        ]
        state = {"messages": msgs}
        config = {"configurable": {"thread_id": "th-1"}}

        result = await summerizer(state, config)

        assert "messages" in result
        result_msgs = result["messages"]
        remove_ops = [m for m in result_msgs if isinstance(m, RemoveMessage)]
        summary_ops = [m for m in result_msgs if isinstance(m, SystemMessage)]

        assert len(remove_ops) == 3  # The 3 oldest messages
        assert len(summary_ops) == 1
        assert "Conversation summary: User discussed Python and Docker." in summary_ops[0].content

    @pytest.mark.asyncio
    @patch("src.nodes.main_nodes.get_llm")
    async def test_test_generation_node_single_chunk(self, mock_get_llm):
        dummy_question = Question(
            question_text="What is 2+2?",
            options=Options(option1="1", option2="4", option3="3", option4="5"),
            correct="option2",
            explanation="2+2=4",
            difficulty="easy",
            points=10,
        )
        dummy_schema = Questions_generation_schema(questions=[dummy_question])

        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = dummy_schema

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_get_llm.return_value = mock_llm

        with patch("src.nodes.main_nodes.TEST_PAPER_GENERATION_PROMPT", MagicMock()):
            with patch("src.nodes.main_nodes.TEST_PAPER_GENERATION_PROMPT.__or__", return_value=mock_chain):
                state = {
                    "total_no_of_questions": 1,
                    "level": "easy",
                    "subject_name": "math",
                    "exam_type": "midterm",
                }
                result = await _test_generation_node(state)

                assert "test_paper" in result
                assert len(result["test_paper"].questions) == 1
                assert result["test_paper"].questions[0].correct == "option2"
