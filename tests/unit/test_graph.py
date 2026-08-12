import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.graphs.builder import get_graph
from src.nodes.conditional_nodes import (
    route_entry,
    route_after_orchastrator,
    route_summary_node,
)


def test_get_graph_compilation():
    """Test get_graph compiles the complete state machine with checkpointer and store."""
    get_graph.cache_clear()

    with patch("src.graphs.builder.get_checkpointer", return_value=MemorySaver()), \
         patch("src.graphs.builder.get_store", return_value=InMemoryStore()):

        graph = get_graph()
        assert isinstance(graph, CompiledStateGraph)
        assert "ingestion_node" in graph.nodes
        assert "orchastrator_node" in graph.nodes
        assert "query_generation_node" in graph.nodes
        assert "retreiver_node" in graph.nodes
        assert "chat_node" in graph.nodes


def test_conditional_routers():
    """Test routing functions for entry, orchestrator output, and summarizer decision."""
    # route_entry
    assert route_entry({"file_paths": ["/tmp/a.pdf"]}) == "ingestion_node"
    assert route_entry({"file_paths": []}) == "orchastrator_node"

    # route_after_orchastrator
    assert route_after_orchastrator({"require_db_search": True}) == "query_generation_node"
    assert route_after_orchastrator({"require_db_search": False, "messages": []}) == "chat_node"

    # route_summary_node
    assert route_summary_node({"messages": []}) == "chat_node"
