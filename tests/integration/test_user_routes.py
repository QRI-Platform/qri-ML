import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def test_get_user_conversation(client: TestClient):
    """Test GET /api/v1/user/conversation endpoint."""
    with patch("api.routes.user_routes.load_conversation", AsyncMock(return_value=[])):
        response = client.get(
            "/api/v1/user/conversation",
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []


def test_get_user_long_term_memory(client: TestClient):
    """Test GET /api/v1/user/long_term_memory/ endpoint."""
    with patch("api.routes.user_routes.get_user_long_term_memory", AsyncMock(return_value=[])):
        response = client.get(
            "/api/v1/user/long_term_memory/",
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []


def test_upsert_long_term_memory(client: TestClient):
    """Test POST /api/v1/user/long_term_memory endpoint."""
    with patch("api.routes.user_routes.upsert_long_term_memory", AsyncMock(return_value="preferred_language")):
        response = client.post(
            "/api/v1/user/long_term_memory",
            json={"key": "Preferred Language", "value": "Python"},
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["key"] == "preferred_language"


def test_delete_long_term_memory_key(client: TestClient):
    """Test DELETE /api/v1/user/long_term_memory/{key} endpoint."""
    with patch("api.routes.user_routes.delete_long_term_memory_key", AsyncMock()):
        response = client.delete(
            "/api/v1/user/long_term_memory/preferred_language",
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


def test_delete_pinecone_namespace(client: TestClient):
    """Test DELETE /api/v1/user/pine_cone endpoint."""
    with patch("api.routes.user_routes.delete_pinecone_namespace", AsyncMock(return_value=True)):
        response = client.delete(
            "/api/v1/user/pine_cone",
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


def test_delete_user_conversation(client: TestClient):
    """Test DELETE /api/v1/user/conversation endpoint."""
    with patch("api.routes.user_routes.delete_user_conversation", AsyncMock(return_value=True)):
        response = client.delete(
            "/api/v1/user/conversation",
            headers={"x-user-id": "user123", "x-thread-id": "thread456"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
