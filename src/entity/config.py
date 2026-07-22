from typing import Optional
from dataclasses import dataclass
from src.constants import *


@dataclass
class RetrieverConfig:
    index_name: str = DEFAULT_INDEX_NAME
    embeding_dim: int = EMBEDDING_DIM
    metric: str = METRIC
    cloud: str = CLOUDE_PROVIDER
    region: str = CLOUD_REGION
    model_name: str = EMBEDDING_MODEL_NAME
    k: int = RETRIVER_TOP_K
    namespace: Optional[str] = NAMESPACE