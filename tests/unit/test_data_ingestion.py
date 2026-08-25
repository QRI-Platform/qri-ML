import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document

from src.services.data_ingestion_service import DataIngestion
from src.domain.config_entities import DataIngestionConfig


def test_inject_filename_metadata():
    """Test filename metadata tagging on documents."""
    docs = [Document(page_content="Text content", metadata={})]
    tagged = DataIngestion._inject_filename_metadata(
        documents=docs,
        file_path="/path/to/report.pdf",
        namespace="thread123",
    )
    assert tagged[0].metadata["filename"] == "thread123_report.pdf"


@pytest.mark.asyncio
async def test_chunk_docs():
    """Test splitting long text into smaller document chunks."""
    long_text = "Word " * 300
    docs = [Document(page_content=long_text, metadata={"filename": "test.txt"})]

    chunks = await DataIngestion.chunk_docs(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all("filename" in c.metadata for c in chunks)


@pytest.mark.asyncio
async def test_ingest_pipeline():
    """Test full ingestion workflow with mocked loader and retriever."""
    config = DataIngestionConfig(
        files_path=["/tmp/doc.txt"],
        namespace="thread_abc",
        chunk_size=100,
        chunk_overlap=10,
    )
    mock_retriever = AsyncMock()
    mock_retriever.create_retriever = AsyncMock(return_value=MagicMock())
    mock_retriever.add_documents = AsyncMock()

    ingestion = DataIngestion(data_ingestion_config=config, retriever=mock_retriever)

    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="Ingested sample content", metadata={})]

    with patch.object(ingestion, "get_loader", AsyncMock(return_value=[mock_loader])):
        artifact = await ingestion.ingest()
        assert artifact.retriever == mock_retriever
        mock_retriever.create_retriever.assert_called_once()
        mock_retriever.add_documents.assert_called_once()


def test_pipeline_options_format_configuration():
    """Verify distinct pipeline options for PDF (images/layout off) and IMAGE (formula/layout/OCR on)."""
    from docling.datamodel.base_models import InputFormat

    config = DataIngestionConfig(
        files_path=["/tmp/test.pdf"],
        namespace="test",
        chunk_size=100,
        chunk_overlap=10,
    )
    mock_retriever = MagicMock()
    ingestion = DataIngestion(data_ingestion_config=config, retriever=mock_retriever)

    fmt_opts = ingestion.custom_converter.format_to_options

    # PDF format check
    pdf_opts = fmt_opts[InputFormat.PDF].pipeline_options
    assert pdf_opts.generate_page_images is False
    assert pdf_opts.generate_picture_images is False
    assert pdf_opts.do_picture_classification is False
    assert pdf_opts.do_ocr is False
    assert pdf_opts.do_formula_enrichment is False

    # IMAGE format check
    img_opts = fmt_opts[InputFormat.IMAGE].pipeline_options
    assert img_opts.generate_page_images is True
    assert img_opts.generate_picture_images is True
    assert img_opts.do_picture_classification is True
    assert img_opts.do_ocr is True
    assert img_opts.do_formula_enrichment is True

