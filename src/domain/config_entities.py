from dataclasses import dataclass

from src.core.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLOUD_PROVIDER,
    CLOUD_REGION,
    DEFAULT_INDEX_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    METRIC,
    RETRIEVER_TOP_K,
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
    namespace: str | None = None


@dataclass
class DataIngestionConfig:
    files_path: list[str]
    index_name: str = DEFAULT_INDEX_NAME
    namespace: str | None = None
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
