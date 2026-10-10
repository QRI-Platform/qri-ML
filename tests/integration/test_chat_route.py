from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage

# We can mock get_llm which is used to construct LLM clients
@patch("src.nodes.main_nodes.get_llm")
def test_unauthorized_access(mock_get_llm, client):
    # This shouldn't even trigger the LLM since it gets 401
    response = client.post("/api/v1/graph/chat", json={"message": "Hello"})
    assert response.status_code == 401

@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_llm_response(mock_get_pipeline, client):
    # Mocking the pipeline is much easier and safer for API route tests
    # since LangGraph's astream_events is complex to mock via just get_llm
    mock_pipeline = AsyncMock()
    
    async def mock_initiate(*args, **kwargs):
        # Yield the exact events the SSE stream parser in stream_chat expects
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "chat_node"},
            "data": {
                "chunk": AIMessage(content="This is a mock response from the LLM.", id="123")
            }
        }
        yield {
            "event": "on_chain_end",
            "name": "chat_node",
            "data": {
                "output": {"ai_response": "This is a mock response from the LLM."}
            }
        }
        
    mock_pipeline.initiate = mock_initiate
    mock_get_pipeline.return_value = mock_pipeline
    
    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    response = client.post("/api/v1/graph/chat", json={"message": "Hello"}, headers=headers)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"
    assert "data:" in response.text


@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_chat_renaming_route(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    async def mock_initiate(*args, **kwargs):
        # Yield the exact events the SSE stream parser in stream_chat expects
        
        yield {
            "event": "on_chain_end",
            "name": "title_renamer_node",
            "data": {
                "output": {"title": "This is the mock title test"}
            }
        }

    mock_pipeline.initiate = mock_initiate

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
        
    }
    response = client.get("/api/v1/graph/chat_rename", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"



@patch("api.routes.graph_routes.get_graph_runner_pipeline")
def test_ingest_route(mock_get_pipeline, client):
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    async def mock_initiate(*args, **kwargs):
        yield {
            "event": "on_chain_end",
            "name": "ingest_node",
            "data": {
                "output": {"status": "success"}
            }
        }

    mock_pipeline.initiate = mock_initiate

    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    mock_get_pipeline.return_value = mock_pipeline
    
    # Needs to send files since it uses multer_middleware
    files = {
        "files": ("test.txt", b"dummy content", "text/plain")
    }
    response = client.post("/api/v1/graph/ingest", files=files, headers=headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"