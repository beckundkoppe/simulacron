from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from ollama import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.providers.llama_cpp_python import LlamaCppPythonProvider
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.chat_history.basic_chat_history import BasicChatHistory, BasicChatHistoryStrategy, Roles

from llm.memory.memory import Memory, MemoryType, Role
from llm.memory.simple import SimpleMemory
from llm.runner import LangchainRunner, LlamaCppRunner, Runner
import debug

class Agent(ABC):
    @abstractmethod
    def invoke(self, message: str) -> str: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[Callable]) -> None: ...

    def build(runner: Runner, memory: Optional[Memory] = None):
        if isinstance(runner, LlamaCppRunner):
            agent = LlamaAgent(runner, memory)
        elif isinstance(runner, LangchainRunner):
            agent = LangchainAgent(runner, memory)
        else:
            raise ValueError("Unsupported configuration")
        
        return agent
    
class LlamaAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LlamaCppRunner)
        self.runner = runner
        self.provider = LlamaCppPythonProvider(runner._llm)
        self.agent = LlamaCppAgent(self.provider)
        self._output_settings = None
        self.memory = memory

    def register_tools(self, tools: Sequence[Callable]) -> None:
     """Register the given functions as LLM tools."""
     valid_tools = []
     for t in tools:
         # requires _tool_meta from explicit @tool decoration
         meta = getattr(t, "_tool_meta", None)
         if not meta:
             raise ValueError(f"Tool {t.__name__} must be decorated with @tool.")
         valid_tools.append(t)

     self._output_settings = LlmStructuredOutputSettings.from_functions(
         valid_tools,
         allow_parallel_function_calling=True,
     )

    def invoke(self, message: str) -> str:
        if self.memory is None:
            memory = SimpleMemory()
        else:
            memory = self.memory

        history = BasicChatHistory(
            chat_history_strategy=BasicChatHistoryStrategy.last_k_messages,
            k=4096,
            llm_provider=self.provider
        )

        for m in memory.get_history(self.runner.backend):
            role_str = m["role"].lower()
            msg      = m["content"]

            if role_str == "user":
                r = Roles.user
            elif role_str == "assistant":
                r = Roles.assistant
            elif role_str == "system":
                r = Roles.system
            elif role_str == "tool":
                r = Roles.tool
            else:
                raise ValueError(f"Unknown role: {role_str}")

            history.add_message({"role": r, "content": msg})

        reply = self.agent.get_chat_response(
            message=message,
            chat_history=history,
            structured_output_settings=self._output_settings,
            print_output=False
        )

        memory.add_message(Role.USER, message)
        memory.add_message(Role.ASSISTANT, reply)
        memory.debug_print(is_agent=True)

        return reply

class LangchainAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LangchainRunner)
        self.runner = runner
        self._llm = runner.llm
        self.memory = memory

    def register_tools(self, runner: Runner, tools: Sequence[Callable]) -> None:
        if(isinstance(self.llm, ChatOpenAI)):
            self._tools = [
                # requires _tool_meta from explicit @tool decoration
                Tool.from_function(
                    t,
                    name=t._tool_meta["name"],
                    description=t._tool_meta["description"],
                )
                for t in tools
            ]
            self.llm = runner.llm.bind_tools(self._tools)
        else:
            openai_tools: list[dict] = []
            for t in tools:
                # requires _tool_meta from explicit @tool decoration
                meta = t._tool_meta
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": meta["name"],
                        "description": meta["description"],
                        "parameters": meta.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                })
            
            self._llm = runner._llm.bind_tools(openai_tools)

    def invoke(self, message: str) -> str:
        if self.memory is None:
            memory = SimpleMemory()
        else:
            memory = self.memory

        memory.add_message(Role.USER, message)
        reply = self._llm.invoke(memory.get_history(self.runner.backend)).content
        memory.add_message(Role.ASSISTANT, reply)
        memory.debug_print(is_agent=True)

        return reply