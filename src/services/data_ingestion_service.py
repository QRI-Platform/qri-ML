import os
import sys
import asyncio
from functools import lru_cache
from typing import List, Literal, Iterator, AsyncIterator

import pymupdf
import pymupdf4llm
from PIL import Image
import numpy as np
from rapid_latex_ocr import LatexOCR
import rapid_latex_ocr.main
from rapidocr_onnxruntime import RapidOCR

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions

from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from langfuse import observe
from src.retrievers.pinecone_retriever import Retriever
from src.domain.config_entities import DataIngestionConfig
from src.domain.artifacts import DataIngestionArtifact
from src.core.constants import DATA_INGEST_NUM_OF_WORKERS
from src.core.logger import logger
from src.core.exceptions import MyException


# --- Monkey-patch for NumPy 2.x and array/PIL compatibility in rapid_latex_ocr ---
def _patched_loop_image_resizer(self, img):
    if isinstance(img, np.ndarray):
        pillow_img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pillow_img = img
    else:
        pillow_img = Image.open(img).convert("RGB")

    pad_img = self.pre_pro.pad(pillow_img)
    input_image = self.pre_pro.minmax_size(pad_img).convert("RGB")
    r, w, h = 1, input_image.size[0], input_image.size[1]
    for _ in range(10):
        h = int(h * r)
        final_img, pad_img = self.pre_process(input_image, r, w, h)

        resizer_res = self.image_resizer([final_img.astype(np.float32)])[0]

        argmax_idx = int(np.asarray(np.argmax(resizer_res, axis=-1)).flat[0])
        w = (argmax_idx + 1) * 32
        if w == pad_img.size[0]:
            break

        r = w / pad_img.size[0]
    return final_img

rapid_latex_ocr.main.LatexOCR.loop_image_resizer = _patched_loop_image_resizer


@lru_cache(maxsize=1)
def get_shared_latex_ocr() -> LatexOCR:
    """Return the cached LaTeX OCR model, downloading it on first initialization."""
    logger.info("Initializing shared LaTeX OCR model...")
    model = LatexOCR()
    logger.info("Shared LaTeX OCR model initialized successfully.")
    return model


@lru_cache(maxsize=1)
def get_shared_rapid_ocr() -> RapidOCR:
    """Return the cached RapidOCR engine, downloading it on first initialization."""
    logger.info("Initializing shared RapidOCR model...")
    engine = RapidOCR()
    logger.info("Shared RapidOCR model initialized successfully.")
    return engine


class CustomLoader(BaseLoader):
    """Custom document loader supporting PDFs, raw image OCR, and LaTeX equation parsing."""

    def __init__(
        self,
        file_path: str,
        pdf_extraction_strategy: Literal["native", "markdown"] = "native",
        image_extraction_strategy: Literal["ocr", "latex"] = "ocr",
    ) -> None:
        self.file_path = file_path
        self.pdf_extraction_strategy = pdf_extraction_strategy
        self.image_extraction_strategy = image_extraction_strategy

    def _is_image(self) -> bool:
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        _, ext = os.path.splitext(self.file_path)
        return ext.lower() in image_extensions

    def lazy_load(self) -> Iterator[Document]:
        if self._is_image():
            if self.image_extraction_strategy == "latex":
                model = get_shared_latex_ocr()
                latex_code, _ = model(self.file_path)
                metadata = {"source": self.file_path, "type": "image_latex"}
                yield Document(page_content=latex_code or "", metadata=metadata)
            else:
                engine = get_shared_rapid_ocr()
                result, _ = engine(self.file_path)
                text = "\n".join([line[1] for line in result]) if result else ""
                metadata = {"source": self.file_path, "type": "image_ocr"}
                yield Document(page_content=text, metadata=metadata)
            return

        if self.pdf_extraction_strategy == "markdown":
            md_text = pymupdf4llm.to_markdown(self.file_path)
            metadata = {"source": self.file_path, "type": "pdf_markdown"}
            yield Document(page_content=md_text, metadata=metadata)
        else:
            with pymupdf.open(self.file_path) as doc:
                total_pages = len(doc)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    metadata = {
                        "source": self.file_path,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                    }
                    yield Document(page_content=text, metadata=metadata)

    async def alazy_load(self) -> AsyncIterator[Document]:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, lambda: list(self.lazy_load()))
        for doc in docs:
            yield doc


@lru_cache(maxsize=1)
def get_shared_docling_converter() -> DocumentConverter:
    """Return a cached singleton instance of DocumentConverter."""
    logger.info("Initializing global Docling DocumentConverter singleton...")
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = False
    pdf_pipeline_options.do_table_structure = False
    pdf_pipeline_options.do_picture_classification = False
    pdf_pipeline_options.generate_page_images = False
    pdf_pipeline_options.generate_picture_images = False
    pdf_pipeline_options.do_formula_enrichment = False
    pdf_pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=DATA_INGEST_NUM_OF_WORKERS,
        device="auto",
    )

    image_pipeline_options = PdfPipelineOptions()
    image_pipeline_options.do_ocr = True
    image_pipeline_options.do_formula_enrichment = True
    image_pipeline_options.do_picture_classification = True
    image_pipeline_options.generate_page_images = True
    image_pipeline_options.generate_picture_images = True
    image_pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=DATA_INGEST_NUM_OF_WORKERS,
        device="auto",
    )

    _shared_docling_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_pipeline_options),
        },
    )
    logger.info("Global Docling DocumentConverter initialized successfully.")
    return _shared_docling_converter


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, retriever: Retriever):
        self.data_ingestion_config = data_ingestion_config
        self.retriever = retriever
        self.custom_converter = get_shared_docling_converter()
        logger.debug("DataIngestion initialized for %d files", len(data_ingestion_config.files_path))

    @staticmethod
    def _inject_filename_metadata(documents: List[Document], file_path: str, namespace: str) -> List[Document]:
        """Inject a unique 'filename' key into each document's metadata."""
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
    async def get_loader(self) -> List[CustomLoader]:
        try:
            logger.info("Initializing loaders for %d input files", len(self.data_ingestion_config.files_path))
            loaders = [
                CustomLoader(
                    file_path=fp,
                    pdf_extraction_strategy="native",
                    image_extraction_strategy="ocr",
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
            loop = asyncio.get_running_loop()

            async def load_one(file_path: str, loader: CustomLoader) -> List[Document]:
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