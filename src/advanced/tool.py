from functools import wraps
from inspect import signature

def tool(func):
    """
    Decorator: mark a function as a tool for both llama_cpp_agent and LangChain.
    Requires a docstring on the function.
    """
    if not func.__doc__:
        raise ValueError(
            f"Function {func.__name__} needs a docstring for tool usage."
        )

    tool_name = func.__name__
    tool_desc = func.__doc__.strip()

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # attach metadata for both backends
    wrapper._tool_meta = {
        "name": tool_name,
        "description": tool_desc,
        "signature": str(signature(func)),
    }

    return wrapper
