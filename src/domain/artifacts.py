from dataclasses import dataclass
from typing import Any


@dataclass
class DataIngestionArtifact:
    retriever: Any
