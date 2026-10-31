from unittest.mock import AsyncMock, patch


@patch("api.routes.performance_routes.get_graph_runner_pipeline")
def test_performance_generation_success(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance

    mock_performance_data = {
        "performance_schema": {
            "weak_area": ["Calculus"],
            "strong_area": ["Algebra"],
            "mid_area": ["Geometry"],
            "recommendations": ["Review integration rules"],
            "overall_feedback": "Good performance overall.",
            "total_correct_answers": 4,
            "total_questions": 5,
        },
        "need_performance_generation": False,
    }

    mock_pipeline_instance.initiate_performance_generation = AsyncMock(
        return_value=mock_performance_data
    )

    headers = {
        "x-user-id": "test_student",
    }
    payload = {
        "total_correct_answers": 4,
        "total_questions": 5,
        "solved_test_paper": {
            "questions": [
                {
                    "question": "What is derivative of x^2?",
                    "user_answer": "2x",
                    "correct_answer": "2x",
                    "topic": "Calculus",
                }
            ]
        },
    }

    response = client.post("/api/v1/performance/", json=payload, headers=headers)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    assert response.headers.get("content-type") == "application/json"

    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["performance_schema"]["strong_area"] == ["Algebra"]
    assert data["data"]["performance_schema"]["total_correct_answers"] == 4
    mock_pipeline_instance.initiate_performance_generation.assert_awaited_once_with(
        user_id="test_student",
        total_correct_answers=4,
        total_questions=5,
        solved_test_paper=payload["solved_test_paper"],
    )


def test_performance_generation_unauthorized(client):
    payload = {
        "total_correct_answers": 2,
        "total_questions": 5,
        "solved_test_paper": {},
    }
    response = client.post("/api/v1/performance/", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"]["message"] == "user_id is required"


@patch("api.routes.performance_routes.get_graph_runner_pipeline")
def test_performance_generation_auth_via_query_param(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.initiate_performance_generation = AsyncMock(
        return_value={"performance_schema": {}, "need_performance_generation": False}
    )

    payload = {
        "total_correct_answers": 1,
        "total_questions": 2,
        "solved_test_paper": {},
    }
    response = client.post(
        "/api/v1/performance/?user_id=query_student",
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_pipeline_instance.initiate_performance_generation.assert_awaited_once_with(
        user_id="query_student",
        total_correct_answers=1,
        total_questions=2,
        solved_test_paper={},
    )


@patch("api.routes.performance_routes.get_graph_runner_pipeline")
def test_performance_generation_default_payload(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.initiate_performance_generation = AsyncMock(
        return_value={"performance_schema": {}, "need_performance_generation": False}
    )

    headers = {"x-user-id": "test_student"}
    # Send empty payload to test defaults
    response = client.post("/api/v1/performance/", json={}, headers=headers)
    assert response.status_code == 200
    assert response.json()["success"] is True


@patch("api.routes.performance_routes.get_graph_runner_pipeline")
def test_performance_generation_pipeline_failure_400(mock_get_pipeline, client):
    mock_pipeline_instance = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.initiate_performance_generation = AsyncMock(
        side_effect=RuntimeError("Graph execution failed")
    )

    headers = {"x-user-id": "test_student"}
    payload = {
        "total_correct_answers": 0,
        "total_questions": 5,
        "solved_test_paper": {},
    }
    response = client.post("/api/v1/performance/", json=payload, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"]["success"] is False
    assert "Graph execution failed" in response.json()["detail"]["data"]
