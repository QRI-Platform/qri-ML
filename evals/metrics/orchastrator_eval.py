import sys
from unittest.mock import MagicMock, patch
from src.core.logger import logger
from src.core.exceptions import MyException
from src.domain.state import State
from src.nodes.main_nodes import orchastrator_node
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

async def eval_orchastrator_target(inputs: dict) -> dict:
    """Target function under evaluation wrapping orchastrator_node.

    Args:
        inputs: Input dictionary containing 'question' and optional 'has_documents'.

    Returns:
        Dict containing predicted 'require_db_search' boolean.
    """
    try:
        question = inputs.get("question", "Hello, this is a test message.")
        has_docs = inputs.get("has_documents", False)
        
        logger.info("Executing eval_orchastrator_target | question='%s' | has_documents=%s", question, has_docs)

        # Construct State as a proper TypedDict dictionary
        state: State = {
            "messages": [HumanMessage(content=question)],
            "file_paths": [],
            "require_db_search": False,
            "has_documents": has_docs,
            "queries": [],
            "retreived_results": [],
            "ai_response": None,
        }

        config: RunnableConfig = {"configurable": {"thread_id": "eval_thread_id"}}

        # Mock Pinecone client so orchastrator_node receives expected vector count for test has_documents
        mock_pc = MagicMock()
        mock_index = MagicMock()
        vector_count = 10 if has_docs else 0
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"eval_thread_id": {"vector_count": vector_count}}
        }
        mock_pc.Index.return_value = mock_index

        with patch("src.nodes.main_nodes.get_pinecone_client", return_value=mock_pc):
            result_dict = await orchastrator_node(state, config=config)

        require_db_search = result_dict.get("require_db_search", False)

        logger.info(
            "eval_orchastrator_target output | question='%s' | require_db_search=%s",
            question,
            require_db_search,
        )
        return {"require_db_search": require_db_search}

    except Exception as e:
        logger.error("Error in eval_orchastrator_target: %s", str(e), exc_info=True)
        raise MyException(e, sys)


async def eval_orchastrator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluator function comparing target output against ground truth.

    Args:
        inputs: Input dictionary passed to target.
        outputs: Output dictionary returned by eval_orchastrator_target.
        reference_outputs: Ground truth dictionary from dataset.

    Returns:
        Structured evaluator outcome dict for LangSmith / benchmark framework.
    """
    try:
        question = inputs.get("question", "N/A")
        actual = outputs.get("require_db_search") if isinstance(outputs, dict) else False
        expected = reference_outputs.get("require_db_search") if isinstance(reference_outputs, dict) else False

        is_match = actual == expected
        score = 1.0 if is_match else 0.0

        if is_match:
            logger.info(
                "PASS [Score: %.1f] | Question: '%s' | Expected: %s | Actual: %s",
                score,
                question,
                expected,
                actual,
            )
        else:
            logger.warning(
                "FAIL [Score: %.1f] | Question: '%s' | Expected: %s | Actual: %s",
                score,
                question,
                expected,
                actual,
            )

        return {
            "key": "orchastrator_accuracy",
            "score": score,
            "comment": f"Passed: match={expected}" if is_match else f"Failed: expected {expected}, got {actual}",
        }

    except Exception as e:
        logger.error("Error in eval_orchastrator evaluator: %s", str(e), exc_info=True)
        return {
            "key": "orchastrator_accuracy",
            "score": 0.0,
            "comment": f"Evaluator exception: {str(e)}",
        }