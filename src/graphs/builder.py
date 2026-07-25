import sys
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from src.logger import logger
from src.exception import MyException
from src.models.workflow_models import State
from src.nodes.main_nodes import (
    ingestion_node,
    orchastrator_node,
    query_generation_node,
    retreiver_node,
    chat_node,
)
from src.nodes.advance_nodes import summerizer, thread_manager_node
from src.nodes.conditional_nodes import route_entry, route_after_orchastrator
from src.core.dependencies import get_checkpointer


@lru_cache
def get_graph():
    try:
        logger.info("Building LangGraph workflow")
        workflow = StateGraph(state_schema=State)

        workflow.add_node("thread_manager_node", thread_manager_node)
        workflow.add_node("ingestion_node", ingestion_node)
        workflow.add_node("orchastrator_node", orchastrator_node)
        workflow.add_node("query_generation_node", query_generation_node)
        workflow.add_node("retreiver_node", retreiver_node)
        workflow.add_node("chat_node", chat_node)

        workflow.add_edge(START, "thread_manager_node")

        workflow.add_conditional_edges(
            "thread_manager_node",
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

        workflow.add_edge("query_generation_node", "retreiver_node")
        workflow.add_edge("retreiver_node", "chat_node")
        workflow.add_edge("chat_node", END)

        graph = workflow.compile(checkpointer=get_checkpointer())
        logger.info("LangGraph workflow compiled successfully")


        try:
            graph.get_graph().draw_mermaid_png(output_file_path="graph_visualization.png")
            logger.info("Graph saved successfully!")
        except Exception as e:
            logger.info(f"Failed to generate graph image: {e}")
        return graph
    except Exception as e:
        logger.error("Failed to build LangGraph workflow: %s", str(e))
        raise MyException(e, sys)