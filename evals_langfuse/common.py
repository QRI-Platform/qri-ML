from langfuse import Evaluation as EvaluationResult
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from functools import lru_cache


@lru_cache()
def get_llm(model="nvidia/nemotron-3-ultra-550b-a55b:free"):
    return ChatOpenRouter(
        model=model,
        # temperature=0.0
    )
    # return ChatOllama(
    #     model="qwen2.5-coder:3b",
    #     temperature=0.0
    # )