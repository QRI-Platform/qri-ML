
from dataclasses import dataclass
from pathlib import Path
from src.core.logger import logger

@dataclass
class OrchastratorEvalConfig:
    """Configuration for the Orchastrator evaluation."""
    dataset_name: str = "orchastrator_eval_dataset"
    eval_name: str = "orchastrator_eval"
    dataset_path: str = str((Path(__file__).parent / "data" / "orchastrator_eval_dataset.json").resolve())

    def __post_init__(self):
        logger.info(
            "Initialized OrchastratorEvalConfig: dataset_name='%s', eval_name='%s', dataset_path='%s'",
            self.dataset_name,
            self.eval_name,
            self.dataset_path,
        )

@dataclass
class GeneratorEvalConfig:
    dataset_name: str = "generator_eval_dataset"
    dataset_path: str = "evals/data/generator_eval_dataset.json"
    eval_name: str = "generator_chat_node_eval"
    max_concurrency: int = 2