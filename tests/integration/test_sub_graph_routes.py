from unittest.mock import AsyncMock, patch


@patch("api.routes.sub_graph_routes.get_graph_runner_pipeline")
def test_test_generation_route(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance

    async def mock_initiate_test_generation(*args, **kwargs):
        return {
            "questions": [
                {
                    "question": "What is the capital of France?",
                    "optionA": "Berlin",
                    "optionB": "Paris",
                    "optionC": "Madrid",
                    "optionD": "Rome",
                    "correctAnswer": "Paris",
                    "explanation": "Paris is the capital of France.",
                }
            ]
        }

    mock_pipeline_instance.initiate_test_generation = mock_initiate_test_generation

    headers = {
        "x-user-id": "test_user",
    }
    params = {
        "total_no_of_questions": 3,
        "level": "easy",
        "subject_name": "math",
        "exam_type": "midterm",
    }

    response = client.get("/api/v1/subgraph/test", params=params, headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"

    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "questions" in data["data"]


def test_test_generation_unauthorized(client):
    response = client.get("/api/v1/subgraph/test")
    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "user_id is required"


@patch("api.routes.sub_graph_routes.get_graph_runner_pipeline")
def test_test_generation_auth_via_query_param(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.initiate_test_generation = AsyncMock(return_value={"questions": []})

    response = client.get("/api/v1/subgraph/test?user_id=test_user")
    assert response.status_code == 200
    assert response.json()["success"] is True


@patch("api.routes.sub_graph_routes.get_graph_runner_pipeline")
def test_test_generation_pipeline_failure_400(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.initiate_test_generation = AsyncMock(side_effect=RuntimeError("LLM generation timed out"))

    headers = {"x-user-id": "test_user"}
    response = client.get("/api/v1/subgraph/test", headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False
