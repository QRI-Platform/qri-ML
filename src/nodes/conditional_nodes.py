from src.domain.state import State
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP


def route_entry(state: State) -> str:
    if state.get("file_paths"):
        return "ingestion_node"
    return "orchastrator_node"


def route_after_orchastrator(state: State) -> str:
    if state.get("require_db_search"):
        return "query_generation_node"
    if len(state.get("messages", [])) > NO_OF_LAST_MESSAGES_TO_KEEP:
        return "summary_node"
    return "chat_node"


def route_summary_node(state: State) -> str:
    
    if len(state.get("messages", [])) > NO_OF_LAST_MESSAGES_TO_KEEP:
        return "summary_node"
    return "chat_node"