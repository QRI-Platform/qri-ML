from langchain_ollama import ChatOllama
from functools import lru_cache

@lru_cache()
def get_llm():
    return ChatOllama(
    model = "openai/gpt-oss-120b",
    temperature=0.0
    )