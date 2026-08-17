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


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, retriever: Retriever):
        self.data_ingestion_config = data_ingestion_config
        self.retriever = retriever
        logger.debug("DataIngestion initialized for %d files", len(data_ingestion_config.files_path))

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
    async def get_loader(self) -> List[DoclingLoader]:
        try:
            logger.info("Initializing loaders for %d input files", len(self.data_ingestion_config.files_path))
            loaders = [DoclingLoader(file_path=fp) for fp in self.data_ingestion_config.files_path]
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

            # DoclingLoader.load() is CPU-heavy + synchronous (PDF parsing, OCR,
            # table extraction). Run each loader in a thread-pool executor so the
            # async event loop is never blocked. Multiple files are loaded in
            # parallel via asyncio.gather().
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
