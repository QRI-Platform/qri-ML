import sys
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Ensure project root is in sys.path for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("POSTGRES_SQL_URL", "postgresql://user:pass@localhost:5432/testdb")
os.environ.setdefault("HF_TOKEN", "test-hf-token")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def mock_checkpointer():
    """In-memory LangGraph checkpointer for state persistence without Postgres."""
    return MemorySaver()


@pytest.fixture
def mock_store():
    """In-memory LangGraph BaseStore for long-term memory without Postgres."""
    return InMemoryStore()


@pytest.fixture(autouse=True)
def mock_memory_singletons(mock_checkpointer, mock_store):
    """Globally mock checkpointer and store so graph compilation uses in-memory backends."""
    with patch("src.core.memory.get_checkpointer", return_value=mock_checkpointer), \
         patch("src.core.memory.get_store", return_value=mock_store), \
         patch("src.graphs.builder.get_checkpointer", return_value=mock_checkpointer), \
         patch("src.graphs.builder.get_store", return_value=mock_store):
        yield


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM supporting standard & structured output calls."""
    from src.domain.state import OrchestratorOutput, OrchastratorOutput, QueryGenerationOutput

    llm = MagicMock()

    async def mock_ainvoke(prompt_input, **kwargs):
        return AIMessage(content='{"response": "Mocked LLM answer", "memory_key": null, "memory_value": null}')

    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    def with_structured_output(schema, method=None):
        mock_struct = MagicMock()
        if schema in (OrchestratorOutput, OrchastratorOutput):
            mock_struct.ainvoke = AsyncMock(return_value=OrchestratorOutput(require_db_search=True))
        elif schema == QueryGenerationOutput:
            mock_struct.ainvoke = AsyncMock(return_value=QueryGenerationOutput(queries=["query 1", "query 2"]))
        else:
            mock_struct.ainvoke = AsyncMock(return_value=MagicMock())
        return mock_struct

    llm.with_structured_output = MagicMock(side_effect=with_structured_output)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_pinecone_client():
    """Mock Pinecone client & Index operations without external network calls."""
    pc = MagicMock()
    pc.has_index.return_value = True

    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = {
        "namespaces": {
            "thread123": {"vector_count": 5},
            "thread456": {"vector_count": 10},
        }
    }
    mock_index.delete.return_value = {}
    pc.Index.return_value = mock_index
    pc.create_index.return_value = {}
    return pc


@pytest.fixture
def mock_embeddings():
    """Mock sentence transformer embeddings loader."""
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 128]
    emb.embed_query.return_value = [0.1] * 128
    return emb


@pytest.fixture
def client(mock_checkpointer, mock_store):
    """FastAPI TestClient with all DB pool and graph singletons mocked."""
    with patch("api.main.connection_pool", new_callable=AsyncMock), \
         patch("api.main.close_connection_pool", new_callable=AsyncMock), \
         patch("api.main.warmup_dependencies"), \
         patch("src.core.memory.get_checkpointer", return_value=mock_checkpointer), \
         patch("src.core.memory.get_store", return_value=mock_store), \
         patch("src.graphs.builder.get_checkpointer", return_value=mock_checkpointer), \
         patch("src.graphs.builder.get_store", return_value=mock_store):
        from api.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as test_client:
            yield test_client
