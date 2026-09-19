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
    """Save an important, persistent user detail in long-term memory.

    Use this tool when the user explicitly shares a personal fact, preference, background detail,
    or project context that should be remembered in future conversations. Do not save temporary
    questions, one-off instructions, or information that belongs only in the current turn.

    Args:
        memory_key: A concise snake_case identifier such as 'user_role' or 'favorite_framework'.
        memory_value: The specific user detail to remember, such as 'FastAPI developer'.
    """
    try:
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        max_long_term_memory_cap = config.get("configurable", {}).get("metadata", {}).get("plan", {}).get("long_term_memory_cap", 5)
        # Clean and normalize the key
        key_name = memory_key.strip().lower().replace(" ", "_").replace("-", "_")
        cleaned_value = memory_value.strip()

        if not key_name or key_name in ["none", "null", "undefined"]:
            logger.warning("Skipping invalid memory_key: '%s'", memory_key)
            return "Failed: Invalid memory key provided."

        logger.info("Checking no of stored long-term memory for user %s", user_id)
        user_memories = await store.asearch(("user", str(user_id), "details"))

        if len(user_memories) >= max_long_term_memory_cap and key_name not in user_memories:
            logger.warning("User %s has reached the long-term memory cap of %d", user_id, max_long_term_memory_cap)
            return f"Failed: Long-term memory cap of {max_long_term_memory_cap} reached. Please delete some memories before adding new ones."
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