from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
import inspect
import json
import re
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
from llm.model import AgentBackend, Model, SourceOllama, SourceRemote
from llm.prepare import prepare_model_source
from llm.provider import LangchainLocalProvider, LangchainRemoteProvider, LlamaCppProvider, Provider
from util import console

class ToolProvider(Provider):
    def _init(self, name: str, model: Model, memory: Optional[Memory] = None) -> None:
        super()._init(name, model, memory)

        self.instance = None

    @abstractmethod
    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[Callable] = None) -> None: ...

    def build(name: str, model: Model, memory: Optional[Memory] = None) -> "ToolProvider":
        if model.value.agent_backend is AgentBackend.LLAMACPPAGENT:
            toolprovider = LlamaToolprovider(name, model, memory)
        elif model.value.agent_backend is AgentBackend.LANGCHAIN:
            return LangchainToolprovider(name, model=model, memory=memory)
        else:
            raise ValueError("Unsupported ToolProvider configuration")

        return toolprovider
    
class LlamaToolprovider(ToolProvider):
    def __init__(self, name: str,  model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)

        if isinstance(model.value.source, SourceRemote):
            raise ValueError("Remote sources are not supported with LLAMACPPAGENT. Use LANGCHAIN backend instead.")

        LlamaCppProvider._init(self, name, model, memory)

        self.instance = LlamaCppAgent(LlamaCppPythonProvider(self.llm), debug_output=debug.VERBOSE_LLAMACPPAGENT)
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
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

        history = BasicChatHistory(
            chat_history_strategy=BasicChatHistoryStrategy.last_k_messages,
            llm_provider=self.llm
        )

        for m in temp:
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
            #system_prompt=transient,
            structured_output_settings=self._output_settings,
            print_output=debug.VERBOSE_LLAMACPPAGENT,
        )

        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

class LangchainToolprovider(ToolProvider):
    def __init__(self, name: str, model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)

        src = model.value.source

        if isinstance(src, SourceOllama):
            LangchainLocalProvider._init(self, name, model, memory)
        elif isinstance(src, SourceRemote):
            LangchainRemoteProvider._init(self, name, model, memory)
        else:
            raise ValueError("Unsupported LangchainToolProvider configuration")

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
            self.instance = self.llm.bind_tools(general_tools)
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
            
            self.instance = self.llm.bind_tools(openai_tools)

    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

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
            
        def _try_execute(raw: str, heuristic: bool = False) -> bool:
            try:
                tc = ToolCall.from_raw(raw, heuristic)
                _execute_toolcall(tc)
                return True
            except Exception as e:
                FormalError(f"toolcall failed: {str(e)}\n{raw}" + """For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}""")
                return False

        #-------------------------

        try:
            result = self.instance.invoke(temp)
        except Exception as e:
            FormalError(f"no valid reply: " + str(e))
            return ""

        raw_reply = result.content

        if debug.VERBOSE_LANGCHAIN_TOOL: console.json_dump(result)

        clean_reply = Provider._clean_reply(raw_reply)

        if self._tools is not None:
            valid_toolcall = False
            for raw in result.tool_calls:
                if _try_execute(raw) is True:
                    valid_toolcall = True

            if(not valid_toolcall):
                parsing_options = [ ast.literal_eval, json.loads, _parse_flexible_json, _parse_call_syntax, _parse_heuristic_1 ]

                for parse in parsing_options:
                    try:
                        data = parse(clean_reply)
                        break
                    except:
                        data = None
                        continue

                if data is not None:
                    _try_execute(data, True)

        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

#interact_with_object_using_item, args: {'object_id': 'door_6', 'using_id': 'key_4'}
def _parse_heuristic_1(cmd: str):
    # Extract name and argument section
    name, rest = cmd.split(",", 1)
    name = name.strip()
    
    # Isolate the argument dict
    args_str = rest.split(":", 1)[1].strip()
    if args_str.startswith("{") and args_str.endswith("}"):
        args_str = args_str[1:-1]

    # Split top-level args
    raw_args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == "," and depth == 0:
            raw_args.append("".join(current).strip())
            current = []
        else:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            current.append(ch)
    if current:
        raw_args.append("".join(current).strip())

    # Convert key:value pairs into dicts
    parsed_args = []
    for arg in raw_args:
        key, val = arg.split(":", 1)
        key = key.strip().strip("'\"")
        val = val.strip().strip("'\"")
        parsed_args.append({"name": key, "value": val})

    return [
        {"name": name, "args": parsed_args}
    ]

#{"name":"interact_with_object","arguments":{ "object_id": "door_6", "operator": "GO_THROUGH" } }
#{"name":"interact_with_object","args":{"object_id":"door_6","operator":"GO_THROUGH"}}
def _parse_flexible_json(cmd: str):
    # Normalize single quotes to double quotes for JSON
    normalized = re.sub(r"'", '"', cmd)

    # Parse JSON
    data = json.loads(normalized)

    # Accept both "args" and "arguments"
    arg_container = data.get("args", None)
    if arg_container is None:
        arg_container = data.get("arguments", {})

    # Build list-style args
    args_list = [{"name": k, "value": v} for k, v in arg_container.items()]

    return [{"name": data["name"], "args": args_list}]

#interact_with_object("door_6", "GO_THROUGH")
def _parse_call_syntax(cmd: str):
    # Extract name and argument list
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*", cmd)
    if not match:
        raise ValueError("Invalid function call syntax")
    
    func_name = match.group(1)
    raw_args = match.group(2).strip()

    # Split arguments respecting quotes
    args = []
    current = []
    depth = 0
    in_string = False
    string_char = None

    for ch in raw_args:
        if ch in ("'", '"'):
            if not in_string:
                in_string = True
                string_char = ch
            elif string_char == ch:
                in_string = False
        if ch == "," and not in_string and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())

    # Clean and normalize values
    cleaned = []
    for a in args:
        a = a.strip()
        if (a.startswith('"') and a.endswith('"')) or (a.startswith("'") and a.endswith("'")):
            a = a[1:-1]
        cleaned.append(a)

    # Convert into required structure
    arg_dicts = [{"name": f"arg{i}", "value": val} for i, val in enumerate(cleaned)]

    return [{"name": func_name, "args": arg_dicts}]
