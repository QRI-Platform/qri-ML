from typing import Optional, List
from dataclasses import dataclass
from src.core.constants import (
    DEFAULT_INDEX_NAME,
    EMBEDDING_DIM,
    METRIC,
    CLOUD_PROVIDER,
    CLOUD_REGION,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


@dataclass
class RetrieverConfig:
    index_name: str = DEFAULT_INDEX_NAME
    embeding_dim: int = EMBEDDING_DIM
    metric: str = METRIC
    cloud: str = CLOUD_PROVIDER
    region: str = CLOUD_REGION
    model_name: str = EMBEDDING_MODEL_NAME
    k: int = RETRIEVER_TOP_K
    namespace: Optional[str] = None


@dataclass
class DataIngestionConfig:
    files_path: List[str]
    index_name: str = DEFAULT_INDEX_NAME
    namespace: Optional[str] = None
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
