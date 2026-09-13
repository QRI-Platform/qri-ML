import sys
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.nodes.main_nodes import (
    agent_node,
    ingestion_node,
    orchastrator_node,
    query_generation_node,
    retreiver_node,
    chat_node,
)
from src.nodes.advance_nodes import summerizer, thread_manager_node,tool_limit_check_node
from src.nodes.conditional_nodes import (
    route_entry,
    route_after_orchastrator,
    route_summary_node,
    route_after_limit,
)
from src.core.memory import get_checkpointer, get_store
from langgraph.prebuilt import ToolNode, tools_condition
from src.tools.solver_tool import solver
from src.tools.save_long_term_memory_tool import save_long_term_memory


# @lru_cache
# def get_graph():
#     try:
#         logger.info("Building LangGraph workflow")
#         workflow = StateGraph(state_schema=State)

#         workflow.add_node("thread_manager_node", thread_manager_node)
#         workflow.add_node("ingestion_node", ingestion_node)
#         workflow.add_node("orchastrator_node", orchastrator_node)
#         workflow.add_node("query_generation_node", query_generation_node)
#         workflow.add_node("retreiver_node", retreiver_node)
#         workflow.add_node("chat_node", chat_node)
#         workflow.add_node("summary_node", summerizer)
#         workflow.add_node("tool_limit_node",tool_limit_check_node)
#         workflow.add_node("tool_node", ToolNode([solver, save_long_term_memory]))
#         workflow.add_edge(START, "thread_manager_node")

#         workflow.add_conditional_edges(
#             "thread_manager_node",
#             route_entry,
#             {
#                 "ingestion_node": "ingestion_node",
#                 "orchastrator_node": "orchastrator_node",
#             },
#         )

#         workflow.add_edge("ingestion_node", END)

#         workflow.add_conditional_edges(
#             "orchastrator_node",
#             route_after_orchastrator,
#             {
#                 "query_generation_node": "query_generation_node",
#                 "summary_node": "summary_node",
#                 "chat_node": "chat_node",
#             },
#         )

#         workflow.add_edge("query_generation_node", "retreiver_node")
#         workflow.add_conditional_edges("retreiver_node", route_summary_node, {
#             "summary_node": "summary_node",
#             "chat_node": "chat_node"
#         })
#         workflow.add_edge("summary_node", "chat_node")
#         workflow.add_conditional_edges("tool_limit_node",route_after_limit,{
#             "tools":"tool_node",
#             END:END
#         })
#         workflow.add_conditional_edges("chat_node", tools_condition, {
#             "tools": "tool_limit_node",
#             END: END
#         })
#         workflow.add_edge("tool_node", "chat_node")

#         graph = workflow.compile(checkpointer=get_checkpointer(), store=get_store())
#         logger.info("LangGraph workflow compiled successfully")

#         try:
#             graph.get_graph().draw_mermaid_png(output_file_path="graph_visualization.png")
#             logger.info("Graph saved successfully!")
#         except Exception as e:
#             logger.info(f"Failed to generate graph image: {e}")
#         return graph
#     except Exception as e:
#         logger.error("Failed to build LangGraph workflow: %s", str(e))
#         raise MyException(e, sys)





@lru_cache
def get_graph():
    try:
        logger.info("Building LangGraph workflow")
        workflow = StateGraph(state_schema=State)

        workflow.add_node("agent_node", agent_node)
        workflow.add_node("summary_node", summerizer)
        workflow.add_node("ingestion_node", ingestion_node)
        workflow.add_conditional_edges(START, route_entry, {
            "summary_node": "summary_node",
            "ingestion_node": "ingestion_node"
        })
        workflow.add_edge("summary_node", "agent_node")
        workflow.add_edge("agent_node", END)
        workflow.add_edge("ingestion_node", END)

        graph = workflow.compile(checkpointer=get_checkpointer(), store=get_store())
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