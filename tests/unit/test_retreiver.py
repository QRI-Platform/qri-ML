import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.retrievers.pinecone_retriever import Retriever, get_retriever, _verified_indexes
from src.domain.config_entities import RetrieverConfig
from src.core.exceptions import MyException


class TestRetrieverUnit:

    def test_sanitize_metadata_preserves_primitives_and_string_lists(self, sample_retriever_config):
        docs = [
            Document(
                page_content="Test content 1",
                metadata={
                    "str_key": "sample_string",
                    "int_key": 42,
                    "float_key": 3.14,
                    "bool_key": True,
                    "list_str_key": ["tag1", "tag2", "tag3"],
                },
            )
        ]
        sanitized = Retriever._sanitize_metadata(docs)
        meta = sanitized[0].metadata
        assert meta["str_key"] == "sample_string"
        assert meta["int_key"] == 42
        assert meta["float_key"] == 3.14
        assert meta["bool_key"] is True
        assert meta["list_str_key"] == ["tag1", "tag2", "tag3"]

    def test_sanitize_metadata_serializes_complex_and_nested_types(self, sample_retriever_config):
        docs = [
            Document(
                page_content="Complex metadata doc",
                metadata={
                    "dict_key": {"sub_key": "val", "count": 10},
                    "list_of_ints": [1, 2, 3],
                    "none_key": None,
                },
            )
        ]
        sanitized = Retriever._sanitize_metadata(docs)
        meta = sanitized[0].metadata
        assert meta["dict_key"] == json.dumps({"sub_key": "val", "count": 10})
        assert meta["list_of_ints"] == json.dumps([1, 2, 3])
        assert meta["none_key"] == "null"

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.PineconeVectorStore")
    @patch("src.retrievers.pinecone_retriever.get_app_config")
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_create_retriever_when_index_exists(
        self, mock_get_client, mock_get_embeddings, mock_get_cfg, mock_vector_store_cls, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_pc.has_index.return_value = True
        mock_get_client.return_value = mock_pc

        cfg = MagicMock()
        cfg.pine_cone_api_key = "fake-key"
        mock_get_cfg.return_value = cfg

        _verified_indexes.clear()

        retriever = Retriever(retriever_config=sample_retriever_config)
        vs = await retriever.create_retriever()

        mock_pc.has_index.assert_called_once_with(sample_retriever_config.index_name)
        mock_pc.create_index.assert_not_called()
        mock_vector_store_cls.assert_called_once()
        assert sample_retriever_config.index_name in _verified_indexes

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.PineconeVectorStore")
    @patch("src.retrievers.pinecone_retriever.get_app_config")
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_create_retriever_when_index_does_not_exist(
        self, mock_get_client, mock_get_embeddings, mock_get_cfg, mock_vector_store_cls, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_pc.has_index.return_value = False
        mock_get_client.return_value = mock_pc

        cfg = MagicMock()
        cfg.pine_cone_api_key = "fake-key"
        mock_get_cfg.return_value = cfg

        _verified_indexes.clear()

        retriever = Retriever(retriever_config=sample_retriever_config)
        vs = await retriever.create_retriever()

        mock_pc.has_index.assert_called_once_with(sample_retriever_config.index_name)
        mock_pc.create_index.assert_called_once()
        assert sample_retriever_config.index_name in _verified_indexes

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_create_retriever_failure_raises_my_exception(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_pc.has_index.side_effect = RuntimeError("Pinecone network error")
        mock_get_client.return_value = mock_pc
        _verified_indexes.clear()

        retriever = Retriever(retriever_config=sample_retriever_config)
        with pytest.raises(MyException):
            await retriever.create_retriever()

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_add_documents(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        mock_vs = MagicMock()

        docs = [Document(page_content="Doc 1", metadata={"key": "val"})]
        await retriever.add_documents(vector_store=mock_vs, documents=docs)

        mock_vs.add_documents.assert_called_once_with(documents=docs)

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_get_similar_documents_without_filter(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Relevant answer", metadata={})
        ]

        results = await retriever.get_similar_documents(
            vector_store=mock_vs, query="What is machine learning?"
        )

        mock_vs.similarity_search.assert_called_once_with(
            query="What is machine learning?",
            k=sample_retriever_config.k,
            filter=None,
        )
        assert len(results) == 1
        assert results[0].page_content == "Relevant answer"

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_get_similar_documents_with_filter(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []

        await retriever.get_similar_documents(
            vector_store=mock_vs,
            query="Search query",
            filter=["thread123_notes.pdf", "thread123_guide.pdf"],
        )

        expected_filter = {"filename": {"$in": ["thread123_notes.pdf", "thread123_guide.pdf"]}}
        mock_vs.similarity_search.assert_called_once_with(
            query="Search query",
            k=sample_retriever_config.k,
            filter=expected_filter,
        )

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_delete_namespace_success(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        result = await retriever.delete_namespace(
            index_name="test-index", namespace="test-ns"
        )

        mock_index.delete.assert_called_once_with(delete_all=True, namespace="test-ns")
        assert result is True

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_delete_namespace_not_found_returns_false(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_index.delete.side_effect = Exception("Namespace not found 404")
        mock_pc.Index.return_value = mock_index
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        result = await retriever.delete_namespace(
            index_name="test-index", namespace="test-ns"
        )

        assert result is False

    @pytest.mark.asyncio
    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    async def test_delete_namespace_unexpected_error_raises_my_exception(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_index.delete.side_effect = RuntimeError("Fatal connection failure")
        mock_pc.Index.return_value = mock_index
        mock_get_client.return_value = mock_pc

        retriever = Retriever(retriever_config=sample_retriever_config)
        with pytest.raises(MyException):
            await retriever.delete_namespace(
                index_name="test-index", namespace="test-ns"
            )

    @patch("src.retrievers.pinecone_retriever.get_embeddings")
    @patch("src.retrievers.pinecone_retriever.get_pinecone_client")
    def test_get_retriever_factory_function(
        self, mock_get_client, mock_get_embeddings, sample_retriever_config
    ):
        retriever = get_retriever(sample_retriever_config)
        assert isinstance(retriever, Retriever)
        assert retriever.retriever_config == sample_retriever_config
