import sys
import math
import numexpr
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from src.core.logger import logger
from src.core.exceptions import MyException


@tool
async def solver(
    expression: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """Calculate a mathematical expression accurately using Python's numexpr engine.

    Use this tool whenever the user asks for arithmetic, powers, trigonometry, logarithms,
    or any other numeric calculation. Do not calculate manually when this tool can provide
    an exact result.

    Args:
        expression: A valid single-line expression such as '2 + 2' or 'sqrt(144) + 5**2'.
            Do not pass assignments, imports, or multi-line code.
    """
    try:
        cleaned_expr = expression.strip().replace("`", "").replace('"', '').replace("'", "")
        logger.info("Executing math solver for expression: '%s'", cleaned_expr)

        local_dict = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
        }

        # numexpr evaluation with restricted global scope for security
        result = numexpr.evaluate(
            cleaned_expr,
            global_dict={},
            local_dict=local_dict,
        )

        return str(result)

    except Exception as e:
        logger.warning("Solver evaluation failed for expression '%s': %s", expression, str(e))
        return f"Error: Unable to evaluate mathematical expression '{expression}'. Please provide a valid arithmetic format."