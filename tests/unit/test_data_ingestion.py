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


def test_fast_file_loader_pypdf():
    """Test FastFileLoader using pypdf parser_type."""
    from src.services.data_ingestion_service import FastFileLoader
    loader = FastFileLoader(file_path="sample.pdf", parser_type="pypdf")
    with patch.object(loader, "_load_pypdf", return_value=[Document(page_content="PDF Fast Content")]):
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "PDF Fast Content"


def test_fast_file_loader_auto_fallback():
    """Test FastFileLoader auto mode falling back to docling when text length is < 50."""
    from src.services.data_ingestion_service import FastFileLoader
    loader = FastFileLoader(file_path="scanned.pdf", parser_type="auto")
    with patch.object(loader, "_load_pypdf", return_value=[Document(page_content="Short")]), \
         patch.object(loader, "_load_docling", return_value=[Document(page_content="OCR Extracted Content")]):
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "OCR Extracted Content"


def test_fast_file_loader_batched_docling():
    """Test FastFileLoader streaming page batches for large PDFs."""
    from src.services.data_ingestion_service import FastFileLoader
    loader = FastFileLoader(file_path="large_scanned.pdf", parser_type="docling", ocr_batch_size=2)
    
    mock_page = MagicMock()
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page, mock_page, mock_page, mock_page, mock_page]  # 5 pages > batch size 2
    
    mock_docling_loader = MagicMock()
    mock_docling_loader.load.return_value = [Document(page_content="Batch Content", metadata={})]

    with patch("src.services.data_ingestion_service.PdfReader", return_value=mock_pdf_reader), \
         patch("src.services.data_ingestion_service.PdfWriter"), \
         patch("src.services.data_ingestion_service.DoclingLoader", return_value=mock_docling_loader), \
         patch("tempfile.NamedTemporaryFile"), \
         patch("os.path.exists", return_value=True), \
         patch("os.remove"):
        docs = loader._load_docling()
        # 5 pages with batch size 2 -> 3 batches -> 3 documents
        assert len(docs) == 3


