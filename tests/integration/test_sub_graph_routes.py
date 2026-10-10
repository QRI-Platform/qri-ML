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
                    "explanation": "Paris is the capital of France."
                }
            ]
        }

    mock_pipeline_instance.initiate_test_generation = mock_initiate_test_generation

    headers = {
        "x-user-id": "test_user",
    }
    
    # Send as query parameters using `params=`
    params = {
        "total_no_of_questions": 3,
        "level": "easy",
        "subject_name": "math",
        "exam_type": "midterm"
    }
    
    response = client.get("/api/v1/subgraph/test", params=params, headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"
    
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert "questions" in data["data"]

