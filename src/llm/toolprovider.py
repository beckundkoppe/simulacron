from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
import inspect
import json
from typing import Any, Callable, Dict, Optional, Sequence

from ollama import Tool
from langchain_community.chat_models import ChatOpenAI
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.providers.llama_cpp_python import LlamaCppPythonProvider
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.chat_history.basic_chat_history import BasicChatHistory, BasicChatHistoryStrategy, Roles

import debug
from enviroment.resultbuffer import FormalError
from llm.memory.memory import Memory, Role
from llm.provider import LangchainLocalProvider, LlamaCppProvider, Provider
from util import console

class ToolProvider(Provider):
    @abstractmethod
    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[Callable] = None) -> None: ...

    def build(provider: Provider) -> "ToolProvider":
        if isinstance(provider, LlamaCppProvider):
            toolprovider = LlamaToolprovider(provider)
        elif isinstance(provider, LangchainLocalProvider):
            toolprovider = LangchainToolprovider(provider)
        else:
            raise ValueError("Unsupported ToolProvider configuration")

        toolprovider.provider = provider

        return toolprovider
    
class LlamaToolprovider(ToolProvider):
    def __init__(self, name: str, provider: Provider, memory: Optional[Memory] = None):
        self.name = name
        self.provider = provider
        self.memory = memory
        self.instance = LlamaCppAgent(LlamaCppPythonProvider(provider.llm), debug_output=debug.VERBOSE_LLAMACPPAGENT)
        self._output_settings = None
        self._tools: Sequence[Callable] = None

    def register_tools(self, tools: Sequence[Callable]) -> None:
        if tools is None:
            self._output_settings = None
            return

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

    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        mem = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

        history = BasicChatHistory(
            chat_history_strategy=BasicChatHistoryStrategy.last_k_messages,
            llm_provider=self.provider
        )

        for m in mem.get_history():
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
            
        reply = self.instance.get_chat_response(
            message=message,
            chat_history=history,
            structured_output_settings=self._output_settings,
            print_output=debug.VERBOSE_LLAMACPPAGENT
        )

        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

class LangchainToolprovider(ToolProvider):
    def __init__(self, name: str, provider: Provider, memory: Optional[Memory] = None):
        self.name = name
        self.provider = provider
        self.memory = memory
        self.instance = provider.llm

    def register_tools(self, tools: Sequence[Callable]) -> None:
        self._tools = tools

        if tools is None:
            self.instance = self.instance.bind_tools([])
            return

        if(isinstance(self.instance, ChatOpenAI)):
            general_tools = [
                # requires _tool_meta from explicit @tool decoration
                Tool.from_function(
                    t,
                    name=t._tool_meta["name"],
                    description=t._tool_meta["description"],
                )
                for t in tools
            ]
            self.instance = self.provider.llm.bind_tools(general_tools)
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
            
            self.instance = self.provider.llm.bind_tools(openai_tools)

    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        mem = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

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
                        raise Exception("wrong key for name")

                    data["id"] = ""
                    data["type"] = "tool_call"

                    if("args" in raw):
                        data["args"] = raw["args"]
                    elif("arguments" in raw):
                        data["args"] = raw["arguments"]
                    elif("parameters" in raw):
                        data["args"] = raw["parameters"]
                    else:
                        raise Exception("wrong key for args")
                else:
                    data = raw

                return cls(**data)
        
        def _execute_toolcall(tool_call: ToolCall) -> str:
            if not self._tools:
                FormalError("toolcall failed: no tools registered for agent.")
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
            
        def _try_execute(raw: str) -> bool:
            try:
                tc = ToolCall.from_raw(raw, False)
                _execute_toolcall(tc)
                return True
            except Exception as e:
                FormalError(f"toolcall failed: {str(e)}\n{raw}" + """For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}""")
                return False

        #-------------------------

        try:
            result = self.instance.invoke(mem.get_history(self.provider.backend))
        except Exception as e:
            FormalError(f"no valid reply")
            return ""

        if debug.VERBOSE_LANGCHAIN: console.json_dump(result)

        clean_reply = Provider._clean_reply(result.content)

        if self._tools is not None:
            valid_toolcall = False
            for raw in result.tool_calls:
                if _try_execute(raw) is True:
                    valid_toolcall = True

            if(not valid_toolcall):
                parsing_options = [ ast.literal_eval, json.loads ]

                for parse in parsing_options:
                    try:
                        data = parse(clean_reply)
                        break
                    except:
                        data = None
                        continue

                if data is not None:
                    _try_execute(data)

        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply