from langfuse import Evaluation as EvaluationResult
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from functools import lru_cache


@lru_cache()
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )
    # return ChatOllama(
    #     model="qwen2.5-coder:3b",
    #     temperature=0.0
    # )