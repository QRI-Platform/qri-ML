import sys
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

from src.core.logger import logger
from src.core.exceptions import MyException


@tool
async def save_long_term_memory(
    memory_key: str,
    memory_value: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Save an important, persistent personal detail, preference, or fact about the user into long-term memory.
    
    Use this tool whenever the user explicitly shares personal information, facts, preferences, 
    or background context (e.g., job role, skills, location, habits, project details) that should be remembered across conversations.
    
    Args:
        memory_key: A concise, descriptive snake_case identifier for the memory (e.g., 'user_role', 'favorite_framework', 'preferred_language').
        memory_value: The specific detail or preference to store (e.g., 'FastAPI developer', 'Ubuntu Linux', 'Python').
    """
    try:
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        
        # Clean and normalize the key
        key_name = memory_key.strip().lower().replace(" ", "_").replace("-", "_")
        cleaned_value = memory_value.strip()

        if not key_name or key_name in ["none", "null", "undefined"]:
            logger.warning("Skipping invalid memory_key: '%s'", memory_key)
            return "Failed: Invalid memory key provided."

        logger.info("Storing long-term memory for user %s: %s = %s", user_id, key_name, cleaned_value)

        # Write directly to LangGraph Store
        await store.aput(
            namespace=("user", str(user_id), "details"),
            key=key_name,
            value={"data": cleaned_value},
        )

        return f"Successfully saved memory: {key_name} = {cleaned_value}"

    except Exception as e:
        logger.error("Failed to save long-term memory: %s", str(e))
        raise MyException(e, sys)