from src.domain.state import State
from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP


def route_entry(state: State) -> str:
    if state.file_paths:
        return "ingestion_node"
    return "orchastrator_node"


def route_after_orchastrator(state: State) -> str:
    if state.require_db_search:
        return "query_generation_node"
    return "chat_node"


def route_summary_node(state: State):
    if len(state.messages) >= NO_OF_LAST_MESSAGES_TO_KEEP:
        return "summary_node"
    return "chat_node"