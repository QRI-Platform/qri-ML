from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage


@patch("src.nodes.main_nodes.get_llm")
def test_unauthorized_access(mock_get_llm, client):
    # Missing both headers and query params should trigger 401
    response = client.post("/api/v1/graph/chat", json={"message": "Hello"})
    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "user_id and thread_id are required"


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_auth_via_query_params(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chat_node"},
            "data": {
                "chunk": AIMessage(content="Hello via query params!", id="123")
            },
        }

    mock_pipeline.initiate = mock_initiate
    mock_get_pipeline.return_value = mock_pipeline

    response = client.post(
        "/api/v1/graph/chat?user_id=query_user&thread_id=query_thread",
        json={"message": "Hi"},
    )
    assert response.status_code == 200
    assert "data:" in response.text
    assert "Hello via query params!" in response.text


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_llm_response(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chat_node"},
            "data": {
                "chunk": AIMessage(content="This is a mock response from the LLM.", id="123")
            },
        }
        yield {
            "event": "on_chain_end",
            "name": "chat_node",
            "data": {
                "output": {"ai_response": "This is a mock response from the LLM."}
            },
        }

    mock_pipeline.initiate = mock_initiate
    mock_get_pipeline.return_value = mock_pipeline

    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.post("/api/v1/graph/chat", json={"message": "Hello"}, headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"
    assert "data:" in response.text
    assert "This is a mock response from the LLM." in response.text


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_tool_events_streaming(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_tool_start",
            "name": "solver",
            "data": {"input": {"expression": "2+2"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "solver",
            "data": {"output": "4"},
        }
        yield {
            "event": "on_chain_end",
            "name": "agent_node",
            "data": {"output": {"ai_response": "The answer is 4."}},
        }

    mock_pipeline.initiate = mock_initiate
    mock_get_pipeline.return_value = mock_pipeline

    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.post("/api/v1/graph/chat", json={"message": "What is 2+2?"}, headers=headers)

    assert response.status_code == 200
    assert "tool_start" in response.text
    assert "tool_end" in response.text
    assert "The answer is 4." in response.text


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_error_streaming(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()

    async def mock_initiate_with_error(*args, **kwargs):
        raise RuntimeError("Pipeline failed abruptly")
        yield  # Make it an async generator

    mock_pipeline.initiate = mock_initiate_with_error
    mock_get_pipeline.return_value = mock_pipeline

    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.post("/api/v1/graph/chat", json={"message": "Hello"}, headers=headers)

    assert response.status_code == 200
    assert "error" in response.text
    assert "Pipeline failed abruptly" in response.text


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_renaming_route(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_chain_end",
            "name": "title_renamer_node",
            "data": {"output": {"title": "This is the mock title test"}},
        }

    mock_pipeline.initiate = mock_initiate

    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.get("/api/v1/graph/chat_rename", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert data["data"]["title"] == "This is the mock title test"


def test_chat_renaming_unauthorized(client):
    response = client.get("/api/v1/graph/chat_rename")
    assert response.status_code == 401


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_renaming_failure_no_title(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    async def mock_initiate_empty(*args, **kwargs):
        # Yield no title
        yield {"event": "on_chain_end", "name": "other_node", "data": {}}

    mock_pipeline.initiate = mock_initiate_empty

    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.get("/api/v1/graph/chat_rename", headers=header)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_ingest_route(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_chain_end",
            "name": "ingest_node",
            "data": {"output": {"status": "success"}},
        }

    mock_pipeline.initiate = mock_initiate

    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    files = {"files": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/api/v1/graph/ingest", files=files, headers=headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Data ingested successfully"


def test_ingest_unauthorized(client):
    files = {"files": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/api/v1/graph/ingest", files=files)
    assert response.status_code == 401


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_ingest_pipeline_error(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()

    async def mock_initiate_error(*args, **kwargs):
        raise ValueError("Ingestion parsing failed")
        yield

    mock_pipeline.initiate = mock_initiate_error
    mock_get_pipeline.return_value = mock_pipeline

    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    files = {"files": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/api/v1/graph/ingest", files=files, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False
