import sys
from typing import List
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.retreiver.retreiver import Retriever
from src.entity.config import DataIngestionConfig
from src.logger import logger
from src.exception import MyException
from src.entity.config import RetrieverConfig
from src.entity.artifact import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig,retreiver_config:RetrieverConfig):
        self.data_ingestion_config = data_ingestion_config
        self.retreiver = Retriever(retriever_config=retreiver_config)

    async def get_loader(self) -> List[DoclingLoader]:
        try:
            logger.info("Initializing loaders for input files.")
            loaders = []
            for file_path in self.data_ingestion_config.files_path:
                loader = DoclingLoader(file_path=file_path)
                loaders.append(loader)
            logger.info(f"Successfully created {len(loaders)} file loaders.")
            return loaders
        except Exception as e:
            logger.error("Error occurred while creating file loaders.")
            raise MyException(e, sys)

    @staticmethod
    async def chunks_docs(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Generated {len(chunked_docs)} chunks.")
            return chunked_docs
        except Exception as e:
            logger.error("Error occurred while chunking documents.")
            raise MyException(e, sys)


    async def save_to_db(self,documents:List[Document]):
        vector_store=await self.retreiver.create_retreiver()
        await self.retreiver.add_documents(vector_store=vector_store,documents=documents)



    async def ingest(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion process.")
            loaders = await self.get_loader()
            all_documents: List[Document] = []

            for loader in loaders:
                docs = loader.load()
                all_documents.extend(docs)

            logger.info(f"Successfully loaded {len(all_documents)} total pages/documents.")

            chunked_documents = await self.chunks_docs(
                documents=all_documents,
                chunk_size=self.data_ingestion_config.chunk_size,
                chunk_overlap=self.data_ingestion_config.chunk_overlap
            )

            logger.info("Data ingestion completed successfully.")
            await self.save_to_db(documents=chunked_documents)

            data_ingestion_artifact = DataIngestionArtifact(retreiver=self.retreiver)
            return data_ingestion_artifact

        except Exception as e:
            logger.error("Error occurred during the ingestion pipeline.")
            raise MyException(e, sys)