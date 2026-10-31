from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.graph import END

from src.core.constants import NO_OF_LAST_MESSAGES_TO_KEEP
from src.graphs.builder import get_graph
from src.nodes.conditional_nodes import (
    route_after_limit,
    route_after_orchastrator,
    route_entry,
    route_summary_node,
)


class TestConditionalRoutingUnit:

    def test_route_entry_with_file_paths(self):
        state = {"file_paths": ["/path/to/doc.pdf"]}
        destination = route_entry(state)
        assert destination == "ingestion_node"

    def test_route_entry_with_need_title(self):
        state = {"need_title": True}
        destination = route_entry(state)
        assert destination == "title_renamer_node"

    def test_route_entry_with_need_test_paper(self):
        state = {"need_test_paper": True}
        destination = route_entry(state)
        assert destination == "test_generation_node"

    def test_route_entry_default_fallback(self):
        state = {}
        destination = route_entry(state)
        assert destination == "summary_node"

    def test_route_entry_precedence_file_paths_over_title(self):
        # file_paths has highest precedence
        state = {"file_paths": ["/path/file.txt"], "need_title": True}
        assert route_entry(state) == "ingestion_node"

    def test_route_after_orchastrator_require_db_search(self):
        state = {"require_db_search": True, "messages": []}
        assert route_after_orchastrator(state) == "query_generation_node"

    def test_route_after_orchastrator_threshold_exceeded(self):
        msgs = [HumanMessage(content=f"msg {i}") for i in range(NO_OF_LAST_MESSAGES_TO_KEEP + 2)]
        state = {"require_db_search": False, "messages": msgs}
        assert route_after_orchastrator(state) == "summary_node"

    def test_route_after_orchastrator_default_to_chat(self):
        msgs = [HumanMessage(content="single message")]
        state = {"require_db_search": False, "messages": msgs}
        assert route_after_orchastrator(state) == "chat_node"

    def test_route_summary_node_exceeding_threshold(self):
        msgs = [HumanMessage(content=f"msg {i}") for i in range(NO_OF_LAST_MESSAGES_TO_KEEP + 1)]
        state = {"messages": msgs}
        assert route_summary_node(state) == "summary_node"

    def test_route_summary_node_within_threshold(self):
        msgs = [HumanMessage(content="msg 1")]
        state = {"messages": msgs}
        assert route_summary_node(state) == "chat_node"

    def test_route_after_limit_when_jump_to_end(self):
        state = {"jump_to": "end"}
        assert route_after_limit(state) == END

    @patch("src.nodes.conditional_nodes.tools_condition")
    def test_route_after_limit_delegates_to_tools_condition(self, mock_tools_condition):
        mock_tools_condition.return_value = "tools"
        state = {"jump_to": None}
        assert route_after_limit(state) == "tools"
        mock_tools_condition.assert_called_once_with(state)


class TestGraphBuilderUnit:

    @patch("src.graphs.builder.get_store")
    @patch("src.graphs.builder.get_checkpointer")
    def test_get_graph_structure_and_nodes(self, mock_checkpointer, mock_store):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        # Clear lru_cache to ensure fresh compilation
        get_graph.cache_clear()
        mock_checkpointer.return_value = InMemorySaver()
        mock_store.return_value = InMemoryStore()

        compiled_graph = get_graph()
        assert compiled_graph is not None

        # Verify nodes are registered in the underlying graph
        node_keys = compiled_graph.nodes.keys()
        expected_nodes = [
            "agent_node",
            "summary_node",
            "ingestion_node",
            "title_renamer_node",
            "test_generation_node",
        ]
        for node in expected_nodes:
            assert node in node_keys, f"Expected node '{node}' not found in graph nodes: {node_keys}"
