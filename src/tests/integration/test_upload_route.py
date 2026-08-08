import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def test_ingest_document_endpoint(client: TestClient, tmp_path):
    """Test POST /api/v1/graph/ingest multipart document upload without external Pinecone calls."""
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("Dummy document text for ingestion.")

    async def mock_ingest_stream(*args, **kwargs):
        yield {"event": "on_chain_start", "data": {}}

    mock_pipeline = MagicMock()
    mock_pipeline.initiate = mock_ingest_stream

    with patch("api.routes.graph_routes.get_graph_runner_pipeline", return_value=mock_pipeline):
        with open(sample_file, "rb") as f:
            response = client.post(
                "/api/v1/graph/ingest",
                files={"files": ("sample.pdf", f, "application/pdf")},
                headers={"x-user-id": "user123", "x-thread-id": "thread456"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Data ingested successfully"
