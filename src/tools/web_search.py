import math
import numexpr
from langchain_core.tools import tool


@tool
def solver(expression: str) -> str:
    """Calculate expression using Python's numexpr library.
    Expression should be a single line mathematical expression."""
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict
        )
    )
