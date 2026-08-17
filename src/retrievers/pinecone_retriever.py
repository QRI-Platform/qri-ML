import sys
import json
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from src.core.logger import logger
from src.core.exceptions import MyException
from src.core.config import get_app_config
from src.retrievers.pinecone_client import get_pinecone_client
from src.embeddings.embedding_loader import get_embeddings
from src.domain.config_entities import RetrieverConfig
from langfuse import observe

_verified_indexes = set()


class Retriever:
    def __init__(self, retriever_config: RetrieverConfig):
        self.retriever_config = retriever_config
        self._pc: Pinecone = get_pinecone_client()
        self._embeddings = get_embeddings()
        logger.debug(
            "Retriever created for namespace=%s index=%s",
            retriever_config.namespace,
            retriever_config.index_name,
        )

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

    @observe(name="pinecone_create_retriever")
    async def create_retriever(self):
        try:
            index_name = self.retriever_config.index_name
            cfg = get_app_config()

            if index_name not in _verified_indexes:
                logger.info("Checking Pinecone index existence for index=%s", index_name)
                if not self._pc.has_index(index_name):
                    logger.info("Index %s not found — creating new Serverless index", index_name)
                    self._pc.create_index(
                        name=index_name,
                        dimension=self.retriever_config.embeding_dim,
                        metric=self.retriever_config.metric,
                        spec=ServerlessSpec(
                            cloud=self.retriever_config.cloud,
                            region=self.retriever_config.region,
                        ),
                    )
                    logger.info("Index %s created successfully", index_name)
                _verified_indexes.add(index_name)

            vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self._embeddings,
                pinecone_api_key=cfg.pine_cone_api_key,
                namespace=self.retriever_config.namespace,
            )
            logger.debug("Pinecone VectorStore initialized for namespace=%s", self.retriever_config.namespace)
            return vector_store
        except Exception as e:
            logger.error("Failed to create vector store retriever")
            raise MyException(e, sys)

    @observe(name="pinecone_add_documents")
    async def add_documents(self, vector_store: PineconeVectorStore, documents: List[Document]):
        try:
            logger.info("Adding %d documents to vector store", len(documents))
            documents = self._sanitize_metadata(documents)
            vector_store.add_documents(documents=documents)
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error("Failed to add documents to vector store")
            raise MyException(e, sys)

    @observe(name="pinecone_similarity_search")
    async def get_similar_documents(
        self,
        vector_store: PineconeVectorStore,
        query: str,
        filter: Optional[List[str]] = None,
    ):
        try:
            logger.info("Similarity search for query: %s", query)

            # Build a proper Pinecone metadata filter when filenames are provided.
            # Each filename in the list is ``{thread_id}_{original_filename}``
            # (as stored during ingestion via _inject_filename_metadata).
            pinecone_filter: Optional[Dict[str, Any]] = None
            if filter:
                pinecone_filter = {"filename": {"$in": filter}}
                logger.info("Applying filename filter: %s", filter)

            results = vector_store.similarity_search(
                query=query,
                k=self.retriever_config.k,
                filter=pinecone_filter,
            )
            logger.info("Retrieved %d documents", len(results))
            return results
        except Exception as e:
            logger.error("Similarity search failed")
            raise MyException(e, sys)

    @observe(name="pinecone_delete_namespace")
    async def delete_namespace(self, index_name: str, namespace: str) -> bool:
        try:
            logger.info("Deleting namespace %s from index %s", namespace, index_name)
            index = self._pc.Index(index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info("Namespace %s deleted successfully", namespace)
            return True
        except Exception as e:
            err_msg = str(e).lower()
            if "not found" in err_msg or "404" in err_msg or "namespace not found" in err_msg:
                logger.warning("Pinecone namespace %s not found (already deleted or never existed)", namespace)
                return False
            logger.error("Failed to delete namespace %s", namespace)
            raise MyException(e, sys)


def get_retriever(retriever_config: RetrieverConfig) -> Retriever:
    return Retriever(retriever_config=retriever_config)
