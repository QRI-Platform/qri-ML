from unittest.mock import patch
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, message_to_dict
from langchain_core.messages import messages_to_dict

# We can mock load_conversation which is used to construct conversation clients

@patch("api.routes.user_routes.load_conversation")
def test_load_conversation_response(mock_load_conversation, client):
    # Langchain's message_to_dict needs to be applied to single messages or use messages_to_dict
    mock_load_conversation.return_value = messages_to_dict([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm good, thank you! How can I assist you today?")
    ])
    
    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    response = client.get("/api/v1/user/conversation", headers=headers)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"


@patch("api.routes.user_routes.get_user_long_term_memory")
def test_get_user_long_term_memory(mock_get_user_long_term_memory, client):
    mock_get_user_long_term_memory.return_value = [
        {"key": "favorite_color", "value": "blue"},
        {"key": "hobby", "value": "reading"},
        {"key": "language", "value": "Python"}
    ]
    
    headers = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    # Note the trailing slash! The router is configured with /long_term_memory/
    response = client.get("/api/v1/user/long_term_memory/", headers=headers)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"


@patch("api.routes.user_routes.delete_pinecone_namespace")
@patch("api.routes.user_routes.delete_has_attributes")
def test_pineconedelete_route(mock_delete_attrs, mock_delete_pinecone_namespace, client):
    mock_delete_pinecone_namespace.return_value = True
    mock_delete_attrs.return_value = True

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    # Route is under /user prefix, not /graph
    response = client.delete("/api/v1/user/pine_cone", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"


@patch("api.routes.user_routes.delete_user_conversation")
def test_user_conversation_delete_route(mock_delete_user_conversation, client):
    mock_delete_user_conversation.return_value = True

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    # Route is under /user prefix
    response = client.delete("/api/v1/user/conversation", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"


@patch("api.routes.user_routes.delete_long_term_memory_key")
def test_user_long_term_memory_single_delete_route(mock_delete_ltm_key, client):
    mock_delete_ltm_key.return_value = None

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    # Provide a valid key name instead of {key}, and correct route prefix
    response = client.delete("/api/v1/user/long_term_memory/preferred_language", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"


@patch("api.routes.user_routes.delete_long_term_memory_key")
@patch("api.routes.user_routes.get_user_long_term_memory")
def test_user_long_term_memory_delete_route(mock_get_ltm, mock_delete_ltm_key, client):
    # Mocking both because the delete_all endpoint first fetches the memories then deletes them one by one
    mock_get_ltm.return_value = [{"key": "lang"}]
    mock_delete_ltm_key.return_value = None

    header = {
        "x-user-id": "test_user",
        "x-thread-id": "test_thread"
    }
    response = client.delete("/api/v1/user/long_term_memory", headers=header)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
