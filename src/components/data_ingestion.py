import sys
from typing import List
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.retreiver.retreiver import Retriever
from src.entity.config import DataIngestionConfig, RetrieverConfig
from src.entity.artifact import DataIngestionArtifact
from src.logger import logger
from src.exception import MyException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, retriever: Retriever):
        self.data_ingestion_config = data_ingestion_config
        self.retriever = retriever
        logger.debug("DataIngestion initialized for %d files", len(data_ingestion_config.files_path))

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

    async def save_to_db(self, documents: List[Document]):
        try:
            logger.info("Saving %d documents to vector store", len(documents))
            vector_store = await self.retriever.create_retriever()
            await self.retriever.add_documents(vector_store=vector_store, documents=documents)
            logger.info("Documents saved to vector store successfully")
        except Exception as e:
            logger.error("Error saving documents to vector store")
            raise MyException(e, sys)

    async def ingest(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion pipeline")
            loaders = await self.get_loader()
            all_documents: List[Document] = []

            for loader in loaders:
                docs = loader.load()
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