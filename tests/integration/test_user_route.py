from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, messages_to_dict


# ── GET /api/v1/user/conversation ───────────────────────────────────────────

@patch("api.routes.user_routes.load_conversation")
def test_load_conversation_response(mock_load_conversation, client):
    mock_load_conversation.return_value = messages_to_dict([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm good, thank you! How can I assist you today?"),
    ])

    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.get("/api/v1/user/conversation", headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 3


def test_load_conversation_unauthorized(client):
    response = client.get("/api/v1/user/conversation")
    assert response.status_code == 401


@patch("api.routes.user_routes.load_conversation")
def test_load_conversation_exception_400(mock_load_conversation, client):
    mock_load_conversation.side_effect = RuntimeError("Database connection failed")
    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.get("/api/v1/user/conversation", headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── GET /api/v1/user/long_term_memory/ ───────────────────────────────────────

@patch("api.routes.user_routes.get_user_long_term_memory")
def test_get_user_long_term_memory(mock_get_user_long_term_memory, client):
    mock_get_user_long_term_memory.return_value = [
        {"key": "favorite_color", "value": "blue"},
        {"key": "hobby", "value": "reading"},
        {"key": "language", "value": "Python"},
    ]

    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.get("/api/v1/user/long_term_memory/", headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 3


def test_get_user_long_term_memory_unauthorized(client):
    response = client.get("/api/v1/user/long_term_memory/")
    assert response.status_code == 401


@patch("api.routes.user_routes.get_user_long_term_memory")
def test_get_user_long_term_memory_exception_400(mock_get_user_long_term_memory, client):
    mock_get_user_long_term_memory.side_effect = RuntimeError("BaseStore lookup failed")
    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.get("/api/v1/user/long_term_memory/", headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── DELETE /api/v1/user/pine_cone ───────────────────────────────────────────

@patch("api.routes.user_routes.delete_pinecone_namespace")
@patch("api.routes.user_routes.delete_has_attributes")
def test_pineconedelete_route(mock_delete_attrs, mock_delete_pinecone_namespace, client):
    mock_delete_pinecone_namespace.return_value = True
    mock_delete_attrs.return_value = True

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.delete("/api/v1/user/pine_cone", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True


@patch("api.routes.user_routes.delete_pinecone_namespace")
@patch("api.routes.user_routes.delete_has_attributes")
def test_pinecone_delete_not_found_404(mock_delete_attrs, mock_delete_pinecone_namespace, client):
    mock_delete_pinecone_namespace.return_value = False
    mock_delete_attrs.return_value = False

    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/pine_cone", headers=header)
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert "Namespace not found" in data["message"]


def test_pinecone_delete_unauthorized(client):
    response = client.delete("/api/v1/user/pine_cone")
    assert response.status_code == 401


@patch("api.routes.user_routes.delete_pinecone_namespace")
def test_pinecone_delete_exception_400(mock_delete_pinecone_namespace, client):
    mock_delete_pinecone_namespace.side_effect = RuntimeError("Pinecone timeout")
    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/pine_cone", headers=header)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── DELETE /api/v1/user/conversation ────────────────────────────────────────

@patch("api.routes.user_routes.delete_user_conversation")
def test_user_conversation_delete_route(mock_delete_user_conversation, client):
    mock_delete_user_conversation.return_value = True

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.delete("/api/v1/user/conversation", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Conversation deleted"


@patch("api.routes.user_routes.delete_user_conversation")
def test_user_conversation_delete_not_found(mock_delete_user_conversation, client):
    mock_delete_user_conversation.return_value = False

    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/conversation", headers=header)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["message"] == "Conversation not found"


def test_user_conversation_delete_unauthorized(client):
    response = client.delete("/api/v1/user/conversation")
    assert response.status_code == 401


@patch("api.routes.user_routes.delete_user_conversation")
def test_user_conversation_delete_exception_400(mock_delete_user_conversation, client):
    mock_delete_user_conversation.side_effect = RuntimeError("Checkpointer storage error")
    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/conversation", headers=header)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── POST /api/v1/user/long_term_memory ──────────────────────────────────────

@patch("api.routes.user_routes.upsert_long_term_memory")
def test_add_or_update_long_term_memory_success(mock_upsert, client):
    mock_upsert.return_value = "preferred_language"

    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    payload = {"key": "Preferred Language", "value": "Python"}
    response = client.post("/api/v1/user/long_term_memory", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["key"] == "preferred_language"
    assert data["data"]["value"] == "Python"
    mock_upsert.assert_called_once_with(user_id="test_user", key="Preferred Language", value="Python")


def test_add_or_update_long_term_memory_unauthorized(client):
    payload = {"key": "language", "value": "Python"}
    response = client.post("/api/v1/user/long_term_memory", json=payload)
    assert response.status_code == 401


def test_add_or_update_long_term_memory_validation_error(client):
    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    # Missing required field "value"
    response = client.post("/api/v1/user/long_term_memory", json={"key": "lang"}, headers=headers)
    assert response.status_code == 422


@patch("api.routes.user_routes.upsert_long_term_memory")
def test_add_or_update_long_term_memory_exception_400(mock_upsert, client):
    mock_upsert.side_effect = RuntimeError("Failed to upsert into BaseStore")
    headers = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    payload = {"key": "language", "value": "Python"}
    response = client.post("/api/v1/user/long_term_memory", json=payload, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── DELETE /api/v1/user/long_term_memory/{key} ──────────────────────────────

@patch("api.routes.user_routes.delete_long_term_memory_key")
def test_user_long_term_memory_single_delete_route(mock_delete_ltm_key, client):
    mock_delete_ltm_key.return_value = None

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.delete("/api/v1/user/long_term_memory/preferred_language", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True
    assert "preferred_language" in data["message"]


def test_user_long_term_memory_single_delete_unauthorized(client):
    response = client.delete("/api/v1/user/long_term_memory/fav_food")
    assert response.status_code == 401


@patch("api.routes.user_routes.delete_long_term_memory_key")
def test_user_long_term_memory_single_delete_exception_400(mock_delete_ltm_key, client):
    mock_delete_ltm_key.side_effect = RuntimeError("Delete key failed")
    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/long_term_memory/fav_food", headers=header)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False


# ── DELETE /api/v1/user/long_term_memory ────────────────────────────────────

@patch("api.routes.user_routes.delete_long_term_memory_key")
@patch("api.routes.user_routes.get_user_long_term_memory")
def test_user_long_term_memory_delete_route(mock_get_ltm, mock_delete_ltm_key, client):
    mock_get_ltm.return_value = [{"key": "lang"}]
    mock_delete_ltm_key.return_value = None

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread",
    }
    response = client.delete("/api/v1/user/long_term_memory", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert data["success"] is True


def test_user_long_term_memory_delete_all_unauthorized(client):
    response = client.delete("/api/v1/user/long_term_memory")
    assert response.status_code == 401


@patch("api.routes.user_routes.get_user_long_term_memory")
def test_user_long_term_memory_delete_all_exception_400(mock_get_ltm, client):
    mock_get_ltm.side_effect = RuntimeError("Failed to fetch memories")
    header = {"x-user-id": "test_user", "x-thread-id": "test_thread"}
    response = client.delete("/api/v1/user/long_term_memory", headers=header)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False
