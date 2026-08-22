import sys
import asyncio
from functools import partial
from typing import List
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from src.retrievers.pinecone_retriever import Retriever
from src.domain.config_entities import DataIngestionConfig
from src.domain.artifacts import DataIngestionArtifact
from src.core.logger import logger
from src.core.exceptions import MyException
from langfuse import observe



# ============== Fast & CPU-Optimized Ingestion Loaders ===========
import gc
import tempfile
import multiprocessing
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    RapidOcrOptions # light ocr c++ based
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_docling.loader import ExportType
from pypdf import PdfReader, PdfWriter


class FastFileLoader:
    """Multi-engine document loader for high performance on CPU.
    
    Supports:
    - 'pypdf': Ultra-fast native text extraction (< 1 sec for 200 pages)
    - 'docling': Heavy structural / table / OCR parsing
    - 'auto': Smart hybrid (tries PyPDF first; falls back to Docling if text yield is ~0)
    """

    def __init__(
        self,
        file_path: str,
        parser_type: str = "auto",
        do_ocr: bool = False,
        ocr_batch_size: int = 15,
        converter: DocumentConverter = None,
    ):
        self.file_path = file_path
        self.parser_type = parser_type
        self.do_ocr = do_ocr
        self.ocr_batch_size = ocr_batch_size
        self.converter = converter

    def _load_pypdf(self) -> List[Document]:
        reader = PdfReader(self.file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": self.file_path, "page": i + 1},
                    )
                )
        return docs

    def _get_docling_converter(self, force_ocr: bool = False) -> DocumentConverter:
        if self.converter is not None and not force_ocr:
            return self.converter
        # Limit CPU threads to prevent starving async event loop / breaking sockets
        num_threads = max(1, min(4, multiprocessing.cpu_count()))
        accel = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.CPU
        )
        pipeline_opts = PdfPipelineOptions(accelerator_options=accel)
        should_ocr = self.do_ocr or force_ocr
        pipeline_opts.do_ocr = should_ocr
        if should_ocr:
            pipeline_opts.ocr_options = RapidOcrOptions()
        pipeline_opts.table_structure_options.mode = TableFormerMode.FAST

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_opts),
            }
        )

    def _load_docling(self, force_ocr: bool = False) -> List[Document]:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            try:
                reader = PdfReader(self.file_path)
                total_pages = len(reader.pages)
            except Exception as e:
                logger.warning("Could not read page count for '%s': %s", self.file_path, e)
                total_pages = 0

            # If page count exceeds batch size, perform streaming page-batched ingestion to prevent RAM exhaustion (OOM)
            if total_pages > self.ocr_batch_size:
                logger.info(
                    "Large PDF detected (%d pages). Processing in batches of %d pages to prevent memory spikes.",
                    total_pages,
                    self.ocr_batch_size,
                )
                all_docs: List[Document] = []
                converter = self._get_docling_converter(force_ocr=force_ocr)

                for start_idx in range(0, total_pages, self.ocr_batch_size):
                    end_idx = min(start_idx + self.ocr_batch_size, total_pages)
                    logger.info("Processing Docling batch: pages %d to %d for '%s'", start_idx + 1, end_idx, self.file_path)

                    writer = PdfWriter()
                    for p in range(start_idx, end_idx):
                        writer.add_page(reader.pages[p])

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp_path = tmp.name
                            writer.write(tmp)

                        loader = DoclingLoader(
                            file_path=tmp_path,
                            converter=converter,
                            export_type=ExportType.MARKDOWN,
                        )
                        batch_docs = loader.load()
                        for d in batch_docs:
                            d.metadata["source"] = self.file_path
                            all_docs.append(d)
                    except Exception as batch_err:
                        logger.error("Error processing page batch %d-%d for '%s': %s", start_idx + 1, end_idx, self.file_path, batch_err)
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                        gc.collect()  # Release memory allocations immediately back to OS

                return all_docs

        # Single-pass parsing for non-PDFs or small PDFs (<= batch_size)
        converter = self._get_docling_converter(force_ocr=force_ocr)
        loader = DoclingLoader(
            file_path=self.file_path,
            converter=converter,
            export_type=ExportType.MARKDOWN,
        )
        docs = loader.load()
        gc.collect()
        return docs

    def load(self) -> List[Document]:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext != ".pdf":
            if ext in [".png", ".jpg", ".jpeg"]:
                return self._load_docling(force_ocr=True)
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": self.file_path})]
            except Exception:
                return self._load_docling(force_ocr=True)

        if self.parser_type == "pypdf":
            return self._load_pypdf()
        elif self.parser_type == "docling":
            return self._load_docling()
        else:  # "auto"
            try:
                docs = self._load_pypdf()
                total_text_len = sum(len(d.page_content.strip()) for d in docs)
                if total_text_len >= 50:
                    logger.info(
                        "Fast PyPDF extraction succeeded for '%s' (%d pages, %d chars)",
                        self.file_path,
                        len(docs),
                        total_text_len,
                    )
                    return docs
                logger.info(
                    "PyPDF extracted minimal text from '%s' (%d chars). Falling back to Docling OCR...",
                    self.file_path,
                    total_text_len,
                )
            except Exception as e:
                logger.warning(
                    "PyPDF extraction failed for '%s': %s. Falling back to Docling...",
                    self.file_path,
                    e,
                )
            return self._load_docling(force_ocr=True)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, retriever: Retriever):
        self.data_ingestion_config = data_ingestion_config
        self.retriever = retriever
        logger.debug("DataIngestion initialized for %d files (parser_type=%s)", len(data_ingestion_config.files_path), data_ingestion_config.parser_type)

    @staticmethod
    def _inject_filename_metadata(documents: List[Document], file_path: str, namespace: str) -> List[Document]:
        """Inject a 'filename' key into every document's metadata.

        The stored value is ``{namespace}_{original_filename}`` so that each
        file is uniquely addressable per thread/namespace in Pinecone.
        This allows the retriever to filter by filename when the user
        mentions ``@filename`` in the chat.
        """
        original_name = os.path.basename(file_path)
        tagged_name = f"{namespace}_{original_name}" if namespace else original_name
        for doc in documents:
            doc.metadata["filename"] = tagged_name
        logger.debug(
            "Injected filename='%s' into %d documents from file '%s'",
            tagged_name,
            len(documents),
            file_path,
        )
        return documents

    @observe(name="docling_file_loader")
    async def get_loader(self) -> List[FastFileLoader]:
        try:
            logger.info("Initializing loaders for %d input files (parser_type=%s)", len(self.data_ingestion_config.files_path), self.data_ingestion_config.parser_type)
            loaders = [
                FastFileLoader(
                    file_path=fp,
                    parser_type=self.data_ingestion_config.parser_type,
                    do_ocr=self.data_ingestion_config.do_ocr,
                    ocr_batch_size=self.data_ingestion_config.ocr_batch_size,
                )
                for fp in self.data_ingestion_config.files_path
            ]
            logger.info("Created %d file loaders", len(loaders))
            return loaders
        except Exception as e:
            logger.error("Error creating file loaders")
            raise MyException(e, sys)

    @staticmethod
    @observe(name="document_chunker")
    async def chunk_docs(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        try:
            logger.info("Splitting %d documents into chunks (size=%d, overlap=%d)", len(documents), chunk_size, chunk_overlap)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunked = splitter.split_documents(documents)
            logger.info("Generated %d chunks", len(chunked))
            return chunked
        except Exception as e:
            logger.error("Error chunking documents")
            raise MyException(e, sys)

    @observe(name="save_to_vector_db")
    async def save_to_db(self, documents: List[Document]):
        try:
            logger.info("Saving %d documents to vector store", len(documents))
            vector_store = await self.retriever.create_retriever()
            await self.retriever.add_documents(vector_store=vector_store, documents=documents)
            logger.info("Documents saved to vector store successfully")
        except Exception as e:
            logger.error("Error saving documents to vector store")
            raise MyException(e, sys)

    @observe(name="data_ingestion_pipeline")
    async def ingest(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion pipeline")
            loaders = await self.get_loader()
            all_documents: List[Document] = []

            namespace = self.data_ingestion_config.namespace or ""
            loop = asyncio.get_event_loop()

            async def load_one(file_path: str, loader) -> List[Document]:
                docs = await loop.run_in_executor(None, loader.load)
                return self._inject_filename_metadata(
                    documents=docs,
                    file_path=file_path,
                    namespace=namespace,
                )

            tasks = [
                load_one(fp, loader)
                for fp, loader in zip(self.data_ingestion_config.files_path, loaders)
            ]
            results = await asyncio.gather(*tasks)
            for docs in results:
                all_documents.extend(docs)
                logger.debug("Loaded %d docs from loader", len(docs))

            logger.info("Loaded %d total documents", len(all_documents))

            chunked_documents = await self.chunk_docs(
                documents=all_documents,
                chunk_size=self.data_ingestion_config.chunk_size,
                chunk_overlap=self.data_ingestion_config.chunk_overlap,
            )

            await self.save_to_db(documents=chunked_documents)
            logger.info("Data ingestion completed successfully")
            return DataIngestionArtifact(retriever=self.retriever)
        except Exception as e:
            logger.error("Data ingestion pipeline failed")
            raise MyException(e, sys)

