from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.conversation_service import (
    delete_long_term_memory_key,
    delete_pinecone_namespace,
    delete_thread_data,
    delete_user_conversation,
    get_user_long_term_memory,
    load_conversation,
    upsert_long_term_memory,
)


class TestConversationServiceUnit:

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_retriever")
    async def test_delete_pinecone_namespace_success(self, mock_get_retriever):
        mock_retriever = AsyncMock()
        mock_retriever.delete_namespace.return_value = True
        mock_get_retriever.return_value = mock_retriever

        result = await delete_pinecone_namespace("thread-xyz")
        assert result is True
        mock_retriever.delete_namespace.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_retriever")
    async def test_delete_pinecone_namespace_failure_returns_false(self, mock_get_retriever):
        mock_retriever = AsyncMock()
        mock_retriever.delete_namespace.side_effect = RuntimeError("Network error")
        mock_get_retriever.return_value = mock_retriever

        result = await delete_pinecone_namespace("thread-xyz")
        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.delete_pinecone_namespace")
    async def test_delete_thread_data_calls_cleanup(self, mock_delete_pinecone):
        mock_delete_pinecone.return_value = True
        await delete_thread_data("thread-123", delay_seconds=0)
        mock_delete_pinecone.assert_called_once_with("thread-123")

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_checkpointer")
    async def test_load_conversation_existing_checkpoint(self, mock_get_checkpointer):
        cp = AsyncMock()
        state_mock = MagicMock()
        raw_msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
        state_mock.checkpoint = {"channel_values": {"messages": raw_msgs}}
        cp.aget_tuple.return_value = state_mock
        mock_get_checkpointer.return_value = cp

        result = await load_conversation(thread_id="th-1", user_id="u-1")
        assert len(result) == 2
        assert result[0]["type"] == "human"
        assert result[0]["data"]["content"] == "Hello"
        assert result[1]["type"] == "ai"
        assert result[1]["data"]["content"] == "Hi there!"

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_checkpointer")
    async def test_load_conversation_empty_or_none_checkpoint(self, mock_get_checkpointer):
        cp = AsyncMock()
        cp.aget_tuple.return_value = None
        mock_get_checkpointer.return_value = cp

        result = await load_conversation(thread_id="th-nonexistent", user_id="u-1")
        assert result == []

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_checkpointer")
    async def test_delete_user_conversation_success(self, mock_get_checkpointer):
        cp = AsyncMock()
        state_mock = MagicMock()
        cp.aget_tuple.return_value = state_mock
        mock_get_checkpointer.return_value = cp

        result = await delete_user_conversation(thread_id="th-1", user_id="u-1")
        assert result is True
        cp.adelete_thread.assert_called_once_with(thread_id="th-1")

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_checkpointer")
    async def test_delete_user_conversation_not_found(self, mock_get_checkpointer):
        cp = AsyncMock()
        cp.aget_tuple.return_value = None
        mock_get_checkpointer.return_value = cp

        result = await delete_user_conversation(thread_id="th-1", user_id="u-1")
        assert result is False
        cp.adelete_thread.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_store")
    async def test_get_user_long_term_memory(self, mock_get_store):
        store = AsyncMock()
        item1 = MagicMock(key="pref_lang", value={"data": "Python"})
        item2 = MagicMock(key="topic", value={"data": "Calculus"})
        store.asearch.return_value = [item1, item2]
        mock_get_store.return_value = store

        memories = await get_user_long_term_memory(user_id="u-1")
        assert len(memories) == 2
        assert memories[0]["key"] == "pref_lang"
        assert memories[0]["value"]["data"] == "Python"
        store.asearch.assert_called_once_with(("user", "u-1", "details"))

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_store")
    async def test_delete_long_term_memory_key(self, mock_get_store):
        store = AsyncMock()
        mock_get_store.return_value = store

        await delete_long_term_memory_key(user_id="u-1", key="pref_lang")
        store.adelete.assert_called_once_with(
            namespace=("user", "u-1", "details"),
            key="pref_lang",
        )

    @pytest.mark.asyncio
    @patch("src.services.conversation_service.get_store")
    async def test_upsert_long_term_memory(self, mock_get_store):
        store = AsyncMock()
        mock_get_store.return_value = store

        await upsert_long_term_memory(user_id="u-1", key="User Framework", value="FastAPI")
        store.aput.assert_called_once_with(
            namespace=("user", "u-1", "details"),
            key="user_framework",
            value={"data": "FastAPI"},
        )
