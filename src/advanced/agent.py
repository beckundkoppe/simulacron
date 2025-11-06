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


from enviroment import current
from enviroment.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from debug.settings import VERBOSE_BACKEND
from llm.memory.memory import Memory, Role
from llm.runner import LangchainRunner, LlamaCppRunner, Runner
import debug.console as console

class Agent(ABC):
    @abstractmethod
    def invoke(self, message: str) -> str: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[Callable]) -> None: ...

    def build(runner: Runner, entity, memory: Optional[Memory] = None):
        if isinstance(runner, LlamaCppRunner):
            agent = LlamaAgent(runner, memory)
        elif isinstance(runner, LangchainRunner):
            agent = LangchainAgent(runner, memory)
        else:
            raise ValueError("Unsupported configuration")

        agent.entity = entity

        return agent
    
    def log(self, role: Role):
        pass

    def process_results(self):
        for result in Resultbuffer.buffer:
            if isinstance(result, FormalError):
                msg = "[FORMAL ERROR] " + result.what
                color = console.Color.RED.value
                self.memory.add_message(Role.SYSTEM, msg)
            if isinstance(result, ActionNotPossible):
                msg = "[ACTION FAILURE] " + result.what
                color = console.Color.RED.value
                self.memory.add_message(Role.USER, msg)  
            if isinstance(result, Success):
                msg = "[ACTION] " + result.what
                color = console.Color.YELLOW.value
                self.memory.add_message(Role.USER, msg)

            console.pretty(
                console.bullet(f"[toolcall]\t{msg}", color),
            )
        Resultbuffer.buffer.clear()

    def entity_step(self, tools, observations) -> str:
        current.AGENT = self
    
        self.register_tools(tools)
        reply = self.invoke(observations)

        self.process_results()

        current.AGENT = None

        return reply
    
class LlamaAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LlamaCppRunner)
        self.runner = runner
        self.provider = LlamaCppPythonProvider(runner._llm)
        self.agent = LlamaCppAgent(self.provider)
        self._output_settings = None
        self.memory = memory
        self.tools: Sequence[Callable] = None

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
            memory = Memory()
        else:
            memory = self.memory

        history = BasicChatHistory(
            chat_history_strategy=BasicChatHistoryStrategy.last_k_messages,
            k=4096,
            llm_provider=self.provider
        )
        
        print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(message))!s}", color=console.Color.CYAN))

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
            print_output=VERBOSE_BACKEND
        )

        if not isinstance(reply, str):
            reply = ""

        console.pretty(console.bullet(f"[assistant] {reply}", color=console.Color.YELLOW))

        memory.add_message(Role.USER, message)
        memory.add_message(Role.ASSISTANT, reply)

        return reply

class LangchainAgent(Agent):
    def __init__(self, runner: Runner, memory: Optional[Memory] = None):
        assert isinstance(runner, LangchainRunner)
        self.runner = runner
        self._llm = runner.llm
        self.memory = memory

    def register_tools(self, tools: Sequence[Callable]) -> None:
        self._tools = tools
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
            self._llm = self.runner._llm.bind_tools(general_tools)
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

    def invoke(self, message: str) -> str:
        @dataclass
        class ToolCall:
            name: str
            args: Dict[str, Any]
            id: str
            type: str
            @classmethod
            def from_raw(cls, raw, heuristic) -> "ToolCall":
                """Accept either a JSON string or a dict and return a ToolCall."""
                    
                if(heuristic):
                    data = {}
                    data["name"] = raw["name"]
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
            tool_map: Dict[str, Callable] = {}
            for t in self._tools:
                if not hasattr(t, "_tool_meta"):
                    raise ValueError(f"Tool {t} is missing required _tool_meta attribute.")
                tool_map[t._tool_meta["name"]] = t

            name = tool_call.name
            args = tool_call.args

            #print(f"[TOOLCALL] {name}, args: {args}")

            if name in tool_map:
                sig = inspect.signature(tool_map[name])
                valid_params = set(sig.parameters.keys())
                filtered_args = {k: v for k, v in args.items() if k in valid_params}
            else:
                filtered_args = {}

            if name in tool_map:
                    tool_map[name](**filtered_args)                    
            else:
                raise Exception


        #-------------------------

        if self.memory is None:
            memory = Memory()
        else:
            memory = self.memory

        memory.add_message(Role.USER, message)

        print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(message))!s}", color=console.Color.CYAN))

        result = self._llm.invoke(memory.get_history(self.runner.backend))
        if VERBOSE_BACKEND: console.json_dump(result)

        reply = result.content

        if not isinstance(reply, str):
            reply = ""

        console.pretty(console.bullet(f"[assistant] {reply}", color=console.Color.YELLOW))

        valid_toolcall = False
        for raw in result.tool_calls:
            try:
                tc = ToolCall.from_raw(raw, False)
                ret = _execute_toolcall(tc)
                valid_toolcall = True
            except:
                FormalError(f"toolcall failed: {raw}")
                current.RESULT.harderror_count += 1


        if(not valid_toolcall):
            try:
                data = ast.literal_eval(reply)
            except:
                try:
                    data = json.loads(reply)
                except:
                    data = None

            if data is not None:
                try:
                    tc = ToolCall.from_raw(data, True)
                    ret = _execute_toolcall(tc)
                except:
                    FormalError(f"toolcall failed: {data}")
                    current.RESULT.harderror_count += 1


        memory.add_message(Role.ASSISTANT, reply)

        return reply
    




    
  # def run_tool_calls(self, prompt: str, tools: Sequence[Callable]) -> str:
  #     """
  #     Drive a prompt through the LLM and execute any tool calls it emits.
  #     Each tool must provide a _tool_meta dict with keys "name" and "description".
  #     """
  #     
  #     # Map tool name -> callable
  #     tool_map: Dict[str, Callable] = {}
  #     for t in tools:
  #         if not hasattr(t, "_tool_meta"):
  #             raise ValueError(f"Tool {t} is missing required _tool_meta attribute.")
  #         tool_map[t._tool_meta["name"]] = t
  #     
  #     tool_llm = self._register_tools(tools)
  #     
  #     # First call to the model
  #     result = tool_llm.invoke(prompt)
  #     
  #     max_rounds = 5          # hard safety limit
  #     rounds = 0
  #     previous_calls = None   # to detect repetition
  #     while getattr(result, "tool_calls", None) and rounds < max_rounds:
  #         print(result)
  #         calls = result.tool_calls
  #         if isinstance(calls, dict):    # normalise to list
  #             calls = [calls]
  #         # Stop if the model repeats the exact same calls
  #         if calls == previous_calls:
  #             prompt += (
  #                 "\n[NOTICE] The same tool call was repeated. "
  #                 "Please continue without calling tools again."
  #             )
  #             break
  #         previous_calls = calls
  #         responses = []        # textual outputs of the tools
  #         feedback_parts = []   # structured feedback for the model
  #         for call in calls:
  #             # Extract tool name and arguments
  #             if isinstance(call, dict):
  #                 name = (
  #                     call.get("name")
  #                     or call.get("function", {}).get("name")
  #                     or "unknown"
  #                 )
  #                 args: Dict[str, Any] = (
  #                     call.get("args")
  #                     or call.get("function", {}).get("arguments", {})
  #                     or {}
  #                 )
  #             else:  # fallback if call is a ToolCall object
  #                 name = getattr(call, "name", "unknown")
  #                 args = getattr(call, "arguments", {})
  #             # --- Filter args: keep only those parameters that exist in the function signature ---
  #             if name in tool_map:
  #                 sig = inspect.signature(tool_map[name])
  #                 valid_params = set(sig.parameters.keys())
  #                 filtered_args = {k: v for k, v in args.items() if k in valid_params}
  #             else:
  #                 filtered_args = {}
  #             # Execute the tool and collect its output
  #             if name in tool_map:
  #                 try:
  #                     out = tool_map[name](**filtered_args)
  #                 except Exception as exc:
  #                     out = f"[Tool {name} failed: {exc}]"
  #             else:
  #                 out = f"[Unknown tool: {name}]"
  #             # -------------------------------------------------------------------------------
  #             responses.append(str(out))
  #             feedback_parts.append(
  #                 f"[TOOL RESPONSE] Tool `{name}` executed successfully. Output:\n{out}"
  #             )
  #         # Build feedback message for the model
  #         feedback = "\n".join(feedback_parts)
  #         feedback += "\n[END OF TOOL RESPONSE] You may continue answering the user."
  #         print(feedback)
  #         # Append feedback to the running prompt and re-invoke the model
  #         prompt = f"{prompt}\n{feedback}"
  #         result = tool_llm.invoke(prompt)
  #         rounds += 1
  #     return result.content