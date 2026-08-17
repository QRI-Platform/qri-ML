import sys
from unittest.mock import MagicMock, patch
from evals.common import EvaluationResult
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


async def eval_orchastrator(input: dict = None, output: dict = None, expected_output: dict = None, **kwargs) -> EvaluationResult:
    """Evaluator function comparing target output against ground truth.

    Args:
        input: Input dictionary passed to target.
        output: Output dictionary returned by eval_orchastrator_target.
        expected_output: Ground truth dictionary from dataset.

    Returns:
        Structured EvaluationResult outcome for benchmark framework.
    """
    try:
        inp = input or kwargs.get("inputs", {})
        out = output or kwargs.get("outputs", {})
        exp = expected_output or kwargs.get("reference_outputs", {})

        question = inp.get("question", "N/A")
        actual = out.get("require_db_search") if isinstance(out, dict) else False
        expected = exp.get("require_db_search") if isinstance(exp, dict) else False

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

        return EvaluationResult(
            name="orchastrator_accuracy",
            value=score,
            comment=f"Passed: match={expected}" if is_match else f"Failed: expected {expected}, got {actual}",
        )

    except Exception as e:
        logger.error("Error in eval_orchastrator evaluator: %s", str(e), exc_info=True)
        return EvaluationResult(
            name="orchastrator_accuracy",
            value=0.0,
            comment=f"Evaluator exception: {str(e)}",
        )