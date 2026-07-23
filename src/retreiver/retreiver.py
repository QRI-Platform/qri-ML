import sys
import json
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config.app_config import app_config
from src.exception import MyException
from src.logger import logger


class Retriever:
    @staticmethod
    def _sanitize_metadata(documents: List[Document]) -> List[Document]:
        for doc in documents:
            sanitized = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    sanitized[key] = value
                else:
                    sanitized[key] = json.dumps(value, default=str)
            doc.metadata = sanitized
        return documents

    def __init__(self, retriever_config):
        self.retreiver_config = retriever_config
        self.pc = Pinecone(api_key=app_config.pine_cone_api_key)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.retreiver_config.model_name
        )

    async def create_retreiver(self):
        try:
            logger.info("Initializing Pinecone index creation process.")
            index_name = self.retreiver_config.index_name

            if not self.pc.has_index(index_name):
                logger.info(f"Index {index_name} not found. Creating a new Serverless index.")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.retreiver_config.embeding_dim,
                    metric=self.retreiver_config.metric,
                    spec=ServerlessSpec(
                        cloud=self.retreiver_config.cloud, 
                        region=self.retreiver_config.region
                    ),
                )
                logger.info(f"Index {index_name} successfully created.")

            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embedding_model,
                pinecone_api_key=app_config.pine_cone_api_key,
                namespace=self.retreiver_config.namespace
            )
            
            logger.info("Pinecone Vector Store initialized successfully.")
            return vector_store

        except Exception as e:
            logger.error("Error occurred while creating vector store retriever.")
            raise MyException(e, sys)

    async def add_documents(self, vector_store: PineconeVectorStore, documents: List[Document]):
        try:
            logger.info(f"Adding {len(documents)} documents to vector store.")
            documents = self._sanitize_metadata(documents)
            vector_store.add_documents(documents=documents)
            logger.info("Documents added successfully.")
        except Exception as e:
            logger.error("Error occurred while adding documents.")
            raise MyException(e, sys)

    async def get_similar_product(self, vector_store: PineconeVectorStore, query: str, filter: Optional[Dict[str, Any]] = None):
        try:
            logger.info(f"Performing similarity search for query: {query}")
            results = vector_store.similarity_search(
                query=query,
                k=self.retreiver_config.k,
                filter=filter
            )
            logger.info(f"Retrieved {len(results)} relevant documents.")
            return results
        except Exception as e:
            logger.error("Error occurred while executing similarity search.")
            raise MyException(e, sys)

    async def delete_namespace(self, index_name: str, namespace: str):
        try:
            logger.info(f"Deleting namespace {namespace} from index {index_name}.")
            index = self.pc.Index(index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Namespace {namespace} deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}.")
            raise MyException(e, sys)


Retreiver = Retriever