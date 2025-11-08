from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
import inspect
import json
from typing import Any, Callable, Dict, Optional, Sequence

from advanced.heuristic import parse_toolcall_json, parse_toolcalls_fallback
from ollama import Tool
from langchain_community.chat_models import ChatOpenAI
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.providers.llama_cpp_python import LlamaCppPythonProvider
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.chat_history.basic_chat_history import BasicChatHistory, BasicChatHistoryStrategy, Roles


from enviroment import current
from enviroment.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from debug.settings import VERBOSE_BACKEND
from llm.memory.memory import Memory, Role
from llm.runner import LangchainRunner, LlamaCppRunner, Runner
import debug.console as console

class Agent(ABC):
    @abstractmethod
    def invoke(self, message: str, hint: str = "") -> str: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[Callable]) -> None: ...

    def build(runner: Runner, memory: Optional[Memory] = None, entity = None, name: str = "assistant"):
        if isinstance(runner, LlamaCppRunner):
            agent = LlamaAgent(runner, memory)
        elif isinstance(runner, LangchainRunner):
            agent = LangchainAgent(runner, memory)
        else:
            raise ValueError("Unsupported configuration")

        agent.entity = entity
        agent.name = name

        return agent
    
class LlamaAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LlamaCppRunner)
        self.runner = runner
        self.provider = LlamaCppPythonProvider(runner._llm)
        self.agent = LlamaCppAgent(self.provider)
        self._output_settings = None
        self.memory = memory or runner.new_memory()
        self.tools: Sequence[Callable] = None

    def register_tools(self, tools: Sequence[Callable]) -> None:
        """Register the given functions as LLM tools."""

        if tools is None:
            tools = []

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

    def invoke(self, message: str, hint: str = "") -> str:
        memory = self.memory

        history = BasicChatHistory(
            chat_history_strategy=BasicChatHistoryStrategy.last_k_messages,
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
            message=message + " " + hint,
            chat_history=history,
            structured_output_settings=self._output_settings,
            print_output=VERBOSE_BACKEND
        )

        if not isinstance(reply, str):
            reply = ""

        console.pretty(console.bullet(f"[{self.name}] {reply}", color=console.Color.YELLOW))

        memory.add_message(Role.USER, message)
        memory.add_message(Role.ASSISTANT, reply)

        return reply

class LangchainAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LangchainRunner)
        self.runner = runner
        self._llm = runner.llm
        self.memory = memory or runner.new_memory()
        self._tools = None

    def register_tools(self, tools: Sequence[Callable]) -> None:
        self._tools = tools

        if tools is None:
            self._llm = self.runner.llm.bind_tools([])
            return

        if(isinstance(self._llm, ChatOpenAI)):
            general_tools = [
                # requires _tool_meta from explicit @tool decoration
                Tool.from_function(
                    t,
                    name=t._tool_meta["name"],
                    description=t._tool_meta["description"],
                )
                for t in tools
            ]
            self._llm = self.runner.llm.bind_tools(general_tools)
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
            
            self._llm = self.runner.llm.bind_tools(openai_tools)

    def invoke(self, message: str, hint: str = "") -> str:
        @dataclass
        class ToolCall:
            name: str
            args: Dict[str, Any]
            id: str
            type: str
            @classmethod
            def from_raw(cls, raw, heuristic) -> "ToolCall":  
                if(heuristic):
                    data = {}

                    if("name" in raw):
                        data["name"] = raw["name"]
                    elif("action" in raw):
                        data["name"] = raw["action"]
                    else:
                        console.pretty(
                            console.bullet(f"LLM IS FUCKING STUPID (wrong key for name): {data}", color=console.Color.RED),
                        )

                    data["id"] = ""
                    data["type"] = "tool_call"

                    if("args" in raw):
                        data["args"] = raw["args"]
                    elif("arguments" in raw):
                        data["args"] = raw["arguments"]
                    elif("parameters" in raw):
                        data["args"] = raw["parameters"]
                    else:
                        console.pretty(
                            console.bullet(f"LLM IS FUCKING STUPID (wrong key for args): {data}", color=console.Color.RED),
                        )
                else:
                    data = raw

                return cls(**data)
        
        def _execute_toolcall(tool_call: ToolCall) -> str:
            if not self._tools:
                FormalError(
                    "toolcall failed: no tools registered for agent."
                )
                current.RESULT.harderror_count += 1
                return ""
            
            tool_map: Dict[str, Callable] = {}
            for t in self._tools:
                if not hasattr(t, "_tool_meta"):
                    raise ValueError(f"Tool {t} is missing required _tool_meta attribute.")
                tool_map[t._tool_meta["name"]] = t

            name = tool_call.name
            args = tool_call.args

            print(f"[TOOLCALL] {name}, args: {args}")

            if name in tool_map:
                sig = inspect.signature(tool_map[name])
                valid_params = set(sig.parameters.keys())
                filtered_args = {k: v for k, v in args.items() if k in valid_params}
            else:
                filtered_args = {}

            if name in tool_map:
                    tool_map[name](**filtered_args)                    
            else:
                raise Exception("unknown tool")


        #-------------------------

        memory = self.memory

        memory.add_message(Role.USER, message)
        try:
            result = self._llm.invoke(memory.get_history(self.runner.backend) + [{"role": Role.USER.value, "content": hint}])
        except Exception as e:
            FormalError(f"llm call failed: {str(e)}\nRETRY WITH A VALID ANSWER")
            return

        if VERBOSE_BACKEND: console.json_dump(result)

        reply = result.content

        if not isinstance(reply, str):
            reply = ""

        console.pretty(console.bullet(f"[{self.name}] {reply}", color=console.Color.YELLOW))

        if self._tools is None:
            return

        valid_toolcall = False
        for raw in result.tool_calls:
            try:
                tc = ToolCall.from_raw(raw, False)
                ret = _execute_toolcall(tc)
                valid_toolcall = True
            except Exception as e:
                FormalError(f"toolcall failed: {str(e)}\n{raw}" + """For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}""")
                current.RESULT.harderror_count += 1


        if(not valid_toolcall):
            try:
                data = ast.literal_eval(reply)
            except:
                try:
                    data = json.loads(reply)
                except:
                    try:
                        data = parse_toolcall_json(reply)
                    except:
                        try:
                            data = parse_toolcalls_fallback(reply)
                        except:
                            data = None

            if data is not None:
                try:
                    tc = ToolCall.from_raw(data, True)
                    ret = _execute_toolcall(tc)
                except Exception as e:
                    FormalError(f"toolcall failed: {str(e)}\n{data}"+"""For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}""")
                    current.RESULT.harderror_count += 1


        memory.add_message(Role.ASSISTANT, reply)

        return reply





