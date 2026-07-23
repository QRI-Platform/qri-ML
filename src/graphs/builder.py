from langgraph.graph import StateGraph, START, END

from src.nodes.main_nodes import (
    ingestion_node,
    orchastrator_node,
    query_generation_node,
    retreiver_loader,
    retreiver_node,
    chat_node,
)
from src.models.workflow_models import State


workflow = StateGraph(state_schema=State)

workflow.add_node("ingestion_node", ingestion_node)
workflow.add_node("orchastrator_node", orchastrator_node)
workflow.add_node("query_generation_node", query_generation_node)
workflow.add_node("retreiver_loader", retreiver_loader)
workflow.add_node("retreiver_node", retreiver_node)
workflow.add_node("chat_node", chat_node)


def route_entry(state: State) -> str:
    if state.file_paths:
        return "ingestion_node"
    return "orchastrator_node"


def route_after_orchastrator(state: State) -> str:
    if state.require_db_search:
        return "query_generation_node"
    return "chat_node"


workflow.add_conditional_edges(
    START,
    route_entry,
    {
        "ingestion_node": "ingestion_node",
        "orchastrator_node": "orchastrator_node",
    },
)

workflow.add_edge("ingestion_node", END)

workflow.add_conditional_edges(
    "orchastrator_node",
    route_after_orchastrator,
    {
        "query_generation_node": "query_generation_node",
        "chat_node": "chat_node",
    },
)

workflow.add_edge("query_generation_node", "retreiver_loader")
workflow.add_edge("retreiver_loader", "retreiver_node")
workflow.add_edge("retreiver_node", "chat_node")
workflow.add_edge("chat_node", END)


graph = workflow.compile()