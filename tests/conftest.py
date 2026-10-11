from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient
from api.main import app
from src.retrievers.pinecone_retriever import Retriever
from src.domain.config_entities import RetrieverConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pytest


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_retriever_config():
    return RetrieverConfig(
        index_name="test-index",
        embeding_dim=384,
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        model_name="test-model",
        k=3,
        namespace="test-thread-123",
    )


@pytest.fixture
def mock_pinecone_client():
    client = MagicMock()
    client.has_index.return_value = True
    index_mock = MagicMock()
    client.Index.return_value = index_mock
    return client


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.add_documents.return_value = ["doc-id-1"]
    store.similarity_search.return_value = []
    return store


@pytest.fixture
def mock_base_store():
    store = AsyncMock()
    store.asearch.return_value = []
    store.aput.return_value = None
    store.adelete.return_value = None
    return store


@pytest.fixture
def mock_checkpointer():
    cp = AsyncMock()
    cp.aget_tuple.return_value = None
    cp.adelete_thread.return_value = None
    return cp
 