from typing import Callable, Sequence
from ollama import Tool
from langchain_community.chat_models import ChatOpenAI

from llm.model import AgentBackend
from llm.runner import LangchainRunner, LlamaCppRunner, Runner

from functools import wraps
from langchain_core.tools import StructuredTool
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

class ToolRunner(Runner):
    def __init__(self, runner: Runner, tools: Sequence[Callable]):
        if isinstance(runner, LlamaCppRunner):
            self.backend = AgentBackend.LLAMACPPAGENT
        elif isinstance(runner, LangchainRunner):
            self.backend = AgentBackend.LANGCHAIN
            self._register_tools_langchain(runner, tools)
        else:
            raise ValueError("Unsupported configuration")
    
    def invoke(self, message: str) -> str:
        return self.llm.invoke(message)

