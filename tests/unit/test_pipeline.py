import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage

from src.pipelines.graph_runner_pipeline import (
    GraphRunnerPipeline,
    get_graph_runner_pipeline,
)
from src.core.exceptions import MyException


@pytest.fixture
def mock_graph():
    """Mock LangGraph compiled state graph."""
    graph = MagicMock()

    async def mock_astream_events(state, config, version):
        yield {"event": "on_chain_start", "data": {"input": state}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Hello")}}
        yield {"event": "on_chain_end", "name": "chat_node", "data": {"output": {"ai_response": "Hello world"}}}

    graph.astream_events.side_effect = mock_astream_events
    return graph


@pytest.fixture
def pipeline(mock_graph):
    """Fixture initializing GraphRunnerPipeline with mocked graph."""
    with patch("src.pipelines.graph_runner_pipeline.get_graph", return_value=mock_graph):
        return GraphRunnerPipeline()


def test_pipeline_init(pipeline, mock_graph):
    """Test pipeline initialization."""
    assert pipeline.graph == mock_graph


@pytest.mark.asyncio
async def test_initiate_success(pipeline, mock_graph):
    """Test successful streaming initiate call with message and file paths."""
    events = []
    async for event in pipeline.initiate(
        user_id="user1",
        thread_id="thread1",
        file_paths=["/tmp/doc.pdf"],
        message="Hello AI",
    ):
        events.append(event)

    assert len(events) == 3
    assert events[0]["event"] == "on_chain_start"
    assert events[1]["data"]["chunk"].content == "Hello"

    mock_graph.astream_events.assert_called_once()
    call_args, call_kwargs = mock_graph.astream_events.call_args
    passed_state = call_args[0]

    assert passed_state["file_paths"] == ["/tmp/doc.pdf"]
    assert len(passed_state["messages"]) == 1
    assert isinstance(passed_state["messages"][0], HumanMessage)
    assert passed_state["messages"][0].content == "Hello AI"
    assert call_kwargs["config"] == {"configurable": {"thread_id": "thread1", "user_id": "user1"}}


@pytest.mark.asyncio
async def test_initiate_exception_handling(pipeline, mock_graph):
    """Test exception wrapping in MyException during streaming failure."""
    async def mock_failing_stream(*args, **kwargs):
        raise ValueError("Stream connection broken")
        yield

    mock_graph.astream_events.side_effect = mock_failing_stream

    with pytest.raises(MyException) as exc_info:
        async for _ in pipeline.initiate(user_id="u1", thread_id="t1"):
            pass

    assert "Stream connection broken" in str(exc_info.value)


def test_get_graph_runner_pipeline_lru_cache():
    """Test singleton LRU cache behavior for pipeline factory."""
    with patch("src.pipelines.graph_runner_pipeline.get_graph"):
        get_graph_runner_pipeline.cache_clear()
        p1 = get_graph_runner_pipeline()
        p2 = get_graph_runner_pipeline()
        assert p1 is p2
