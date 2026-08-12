import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_chat_sse_endpoint(client: TestClient):
    """Test POST /api/v1/graph/chat SSE streaming response without any external API calls."""
    async def mock_stream_initiate(*args, **kwargs):
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Hello ")}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="World!")}}
        yield {"event": "on_chain_end", "name": "chat_node", "data": {"output": {"ai_response": "Hello World!"}}}

    mock_pipeline = MagicMock()
    mock_pipeline.initiate = mock_stream_initiate

    with patch("api.routes.graph_routes.get_graph_runner_pipeline", return_value=mock_pipeline):
        response = client.post(
            "/api/v1/graph/chat",
            json={"message": "What is Python?"},
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data:" in response.text
