import os
from functools import lru_cache
from dotenv import load_dotenv
from deepeval.models import OllamaModel

load_dotenv()
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen2.5:1.5b-instruct")


@lru_cache()
def get_judge(model=None):
    return OllamaModel(
        model=model or OLLAMA_JUDGE_MODEL,
        generation_kwargs={"num_predict": 512},
    )