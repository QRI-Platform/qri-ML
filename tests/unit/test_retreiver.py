import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.retrievers.pinecone_retriever import Retriever, get_retriever
from src.domain.config_entities import RetrieverConfig


@pytest.fixture
def retriever_cfg():
    return RetrieverConfig(
        index_name="test_idx",
        namespace="test_ns",
        embeding_dim=128,
        metric="cosine",
        cloud="aws",
        region="us-west-1",
        k=3,
    )


@pytest.fixture
def retriever_inst(retriever_cfg):
    with patch("src.retrievers.pinecone_retriever.get_pinecone_client") as mock_get_pc, \
         patch("src.retrievers.pinecone_retriever.get_embeddings") as mock_get_emb, \
         patch("src.retrievers.pinecone_retriever.get_app_config"):
        
        mock_pc = MagicMock()
        mock_pc.has_index.return_value = False
        mock_get_pc.return_value = mock_pc
        mock_get_emb.return_value = MagicMock()

        r = Retriever(retriever_config=retriever_cfg)
        r._pc = mock_pc
        return r


# 1. Test _sanitize_metadata
def test_sanitize_metadata():
    docs = [
        Document(
            page_content="sample text",
            metadata={"str_key": "val", "int_key": 10, "dict_key": {"a": 1}},
        )
    ]
    sanitized = Retriever._sanitize_metadata(docs)
    assert sanitized[0].metadata["str_key"] == "val"
    assert sanitized[0].metadata["int_key"] == 10
    assert isinstance(sanitized[0].metadata["dict_key"], str)


# 2. Test create_retriever
@pytest.mark.asyncio
@patch("src.retrievers.pinecone_retriever.PineconeVectorStore")
async def test_create_retriever(mock_vs_class, retriever_inst: Retriever):
    store = await retriever_inst.create_retriever()
    retriever_inst._pc.has_index.assert_called_once_with("test_idx")
    retriever_inst._pc.create_index.assert_called_once()
    assert store == mock_vs_class.return_value


# 3. Test add_documents
@pytest.mark.asyncio
async def test_add_documents(retriever_inst: Retriever):
    mock_vector_store = MagicMock()
    docs = [Document(page_content="hello", metadata={"num": 1})]
    
    await retriever_inst.add_documents(
        vector_store=mock_vector_store,
        documents=docs,
    )
    mock_vector_store.add_documents.assert_called_once_with(documents=docs)


# 4. Test get_similar_documents with @filename filter
@pytest.mark.asyncio
async def test_get_similar_documents(retriever_inst: Retriever):
    mock_vector_store = MagicMock()
    expected_docs = [Document(page_content="result")]
    mock_vector_store.similarity_search.return_value = expected_docs

    results = await retriever_inst.get_similar_documents(
        vector_store=mock_vector_store,
        query="search query",
        filter=["file1.pdf", "file2.pdf"],
    )

    mock_vector_store.similarity_search.assert_called_once_with(
        query="search query",
        k=3,
        filter={"filename": {"$in": ["file1.pdf", "file2.pdf"]}},
    )
    assert results == expected_docs


# 5. Test delete_namespace
@pytest.mark.asyncio
async def test_delete_namespace(retriever_inst: Retriever):
    mock_index = MagicMock()
    retriever_inst._pc.Index.return_value = mock_index

    res = await retriever_inst.delete_namespace("test_idx", "test_ns")
    
    retriever_inst._pc.Index.assert_called_once_with("test_idx")
    mock_index.delete.assert_called_once_with(delete_all=True, namespace="test_ns")
    assert res is True


# 6. Test get_retriever factory function
def test_get_retriever_factory(retriever_cfg):
    with patch("src.retrievers.pinecone_retriever.get_pinecone_client"), \
         patch("src.retrievers.pinecone_retriever.get_embeddings"), \
         patch("src.retrievers.pinecone_retriever.get_app_config"):
        r = get_retriever(retriever_cfg)
        assert isinstance(r, Retriever)