from src.models.workflow_models import State


def route_entry(state: State) -> str:
    if state.file_paths:
        return "ingestion_node"
    return "orchastrator_node"


def route_after_orchastrator(state: State) -> str:
    if state.require_db_search:
        return "query_generation_node"
    return "chat_node"
