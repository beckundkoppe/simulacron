from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional
from llama_cpp import Llama
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

import debug
from llm.memory.memory import Memory, Role
from llm.memory.simple import SimpleMemory
from llm.prepare import prepare_model_source
from llm.model import AgentBackend, Backend, Model, SourceFile, SourceHuggingface, SourceLink, SourceOllama, SourceRemote

# ----------------------------------------------------------
# Baseclass
# ----------------------------------------------------------
class Runner(ABC):
    _n_gpu_layers=-1
    _n_ctx=4096
    _n_threads=24
    _max_tokens=4096

    @abstractmethod
    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str: ...

    def build(model: Model) -> "Runner":
        prepare_model_source(model)
        src = model.value.source

        if isinstance(src, SourceOllama):
            return LangchainLocalRunner(model=src.model_id)
        if isinstance(src, SourceRemote):
            raise NotImplementedError
        elif isinstance(src, SourceFile):
            path = src.path
        elif isinstance(src, SourceLink):
            path = src.path
        elif isinstance(src, SourceHuggingface):
            path = src.local_dir + src.filename
        else:
            raise ValueError("Unsupported configuration")

        if model.value.agent_backend is AgentBackend.LLAMACPPAGENT:
            return LlamaCppRunner(path)
        
        raise ValueError("Unsupported configuration")

# ----------------------------------------------------------
# LlamaCppAgent
# ----------------------------------------------------------
class LlamaCppRunner(Runner):
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=Runner._n_gpu_layers,
            n_ctx=Runner._n_ctx,
            n_threads=Runner._n_threads,
            max_tokens=Runner._max_tokens,
        )

        self.backend = Backend.LLAMACPP
    
    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str:
        if memory is None:
            memory = SimpleMemory()
        memory.add_message(role, message)
        reply = self._llm.create_chat_completion(memory.get_history(self.backend))["choices"][0]["message"]["content"]
        memory.add_message(Role.ASSISTANT, reply)
        memory.debug_print()
        return reply

# ----------------------------------------------------------
# LangChainn
# ----------------------------------------------------------
class LangchainRunner(Runner):
    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str:
        if memory is None:
            memory = SimpleMemory()
        memory.add_message(role, message)
        reply = self.llm.invoke(memory.get_history(self.backend)).content
        memory.add_message(Role.ASSISTANT, reply)
        memory.debug_print()
        return reply

    #def invoke(self, prompt: str, tools: Sequence[Callable] = None) -> str:
    #    if tools is None:
    #        return self._llm.invoke(prompt)
    #    else:
    #        tool_llm = self._register_tools(tools)
    #        return tool_llm.invoke(prompt)

    #def invoke(self, prompt: str, tools: Sequence[Callable] | None = None) -> str:
    #    self.add_message("human", prompt)
    #    if tools is None:
    #        reply = self._llm.invoke(self.history())
    #        self.add_message("assistant", reply.content)
    #        return reply.content
    #    else:
    #        #tool_llm = self._register_tools(tools)
    #        #result = tool_llm.invoke(prompt)
    #        #return result
    #        reply = self.run_tool_calls(self.history(), tools)
    #    
    #        #self.add_message("assistant", reply)
    #        return reply
    #    
    #def run_tool_calls(self, prompt: str, tools: Sequence[Callable]) -> str:
    #    """
    #    Drive a prompt through the LLM and execute any tool calls it emits.
    #    Each tool must provide a _tool_meta dict with keys "name" and "description".
    #    """
#
    #    # Map tool name -> callable
    #    tool_map: Dict[str, Callable] = {}
    #    for t in tools:
    #        if not hasattr(t, "_tool_meta"):
    #            raise ValueError(f"Tool {t} is missing required _tool_meta attribute.")
    #        tool_map[t._tool_meta["name"]] = t
#
    #    tool_llm = self._register_tools(tools)
#
    #    # First call to the model
    #    result = tool_llm.invoke(prompt)
#
    #    max_rounds = 5          # hard safety limit
    ##    rounds = 0
    ##    previous_calls = None   # to detect repetition
##
    ##    while getattr(result, "tool_calls", None) and rounds < max_rounds:
    ##        print(result)
    ##        calls = result.tool_calls
    ##        if isinstance(calls, dict):    # normalise to list
    ##            calls = [calls]
##
    ##        # Stop if the model repeats the exact same calls
    ##        if calls == previous_calls:
    ##            prompt += (
    ##                "\n[NOTICE] The same tool call was repeated. "
    ##                "Please continue without calling tools again."
    ##            )
    ##            break
    ##        previous_calls = calls
##
    ##        responses = []        # textual outputs of the tools
    ##        feedback_parts = []   # structured feedback for the model
##
    ##        for call in calls:
    ##            # Extract tool name and arguments
    ##            if isinstance(call, dict):
    ##                name = (
    ##                    call.get("name")
    ##                    or call.get("function", {}).get("name")
    ##                    or "unknown"
    ##                )
    ##                args: Dict[str, Any] = (
    ##                    call.get("args")
    ##                    or call.get("function", {}).get("arguments", {})
    ##                    or {}
    ##                )
    ##            else:  # fallback if call is a ToolCall object
    ##                name = getattr(call, "name", "unknown")
    ##                args = getattr(call, "arguments", {})
##
    ##            # --- Filter args: keep only those parameters that exist in the function signature ---
    ##            if name in tool_map:
    ##                sig = inspect.signature(tool_map[name])
    ##                valid_params = set(sig.parameters.keys())
    ##                filtered_args = {k: v for k, v in args.items() if k in valid_params}
    ##            else:
    ##                filtered_args = {}
##
    ##            # Execute the tool and collect its output
    ##            if name in tool_map:
    ##                try:
    ##                    out = tool_map[name](**filtered_args)
    ##                except Exception as exc:
    ##                    out = f"[Tool {name} failed: {exc}]"
    ##            else:
    ##                out = f"[Unknown tool: {name}]"
    ##            # -------------------------------------------------------------------------------
##
    ##            responses.append(str(out))
    ##            feedback_parts.append(
    ##                f"[TOOL RESPONSE] Tool `{name}` executed successfully. Output:\n{out}"
    ##            )
##
    ##        # Build feedback message for the model
    ##        feedback = "\n".join(feedback_parts)
    ##        feedback += "\n[END OF TOOL RESPONSE] You may continue answering the user."
    ##        print(feedback)
##
    ##        # Append feedback to the running prompt and re-invoke the model
    ##        prompt = f"{prompt}\n{feedback}"
    ##        result = tool_llm.invoke(prompt)
    ##        rounds += 1
##
    ##    return result.content
##

class LangchainLocalRunner(LangchainRunner):
    def __init__(self, model: str):
        """
        model_path:  local path of .GGUF file
        """ 
        self.llm = ChatOllama(model=model)
        self.backend = Backend.OLLAMA

class LangchainRemoteRunner(LangchainRunner):
    def __init__(self, model: str | None = None, api_base: str | None = None):
        """
        model:       e.g "gpt-4" or "gpt-3.5-turbo"
        api_base:    endpoint for OpenAI compatible API
        """
        #self._runner = ChatOpenAI(model=model, openai_api_base=api_base)
        self.backend = Backend.OTHER
        raise NotImplementedError
