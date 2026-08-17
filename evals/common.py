from langchain_ollama import ChatOllama

from functools import lru_cache

@lru_cache()
def get_llm():
    return ChatOllama(
    model = "qwen2.5-coder:3b",
    temperature=0.0
    )