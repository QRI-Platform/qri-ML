from langchain_groq import ChatGroq
from src.config.app_config import app_config

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=app_config.groq_api_key
)