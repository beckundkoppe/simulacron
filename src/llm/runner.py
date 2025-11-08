from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

from gguf import GGUFReader
from llama_cpp import Llama
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

from debug.settings import VERBOSE_BACKEND
from llm.memory.memory import Memory, Role, SummarizingMemory
from llm.prepare import prepare_model_source
from llm.model import (
    AgentBackend,
    Backend,
    Model,
    SourceFile,
    SourceHuggingface,
    SourceLink,
    SourceOllama,
    SourceRemote,
)

# ----------------------------------------------------------
# Baseclass
# ----------------------------------------------------------
class Runner(ABC):
    _n_gpu_layers = -1
    _default_n_ctx = 8192
    _n_threads = 24
    _default_max_tokens = 4096

    def __init__(self, context_window: Optional[int] = None) -> None:
        self.context_window = context_window
        if context_window is not None:
            self.max_tokens = max(256, min(self._default_max_tokens, context_window // 2))
        else:
            self.max_tokens = self._default_max_tokens

    @abstractmethod
    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str: ...

    def build(model: Model) -> "Runner":
        prepare_model_source(model)
        src = model.value.source
        context_window = _resolve_context_window(model)

        if isinstance(src, SourceOllama):
            return LangchainLocalRunner(model=src.model_id, context_window=context_window)
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
            return LlamaCppRunner(path, context_window=context_window)

        raise ValueError("Unsupported configuration")

    def new_memory(self) -> Memory:
        if self.context_window:
            return SummarizingMemory(self.context_window)
        return Memory()

# ----------------------------------------------------------
# LlamaCppAgent
# ----------------------------------------------------------
class LlamaCppRunner(Runner):
    def __init__(self, model_path: str, context_window: Optional[int] = None):
        super().__init__(context_window=context_window)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        n_ctx = self.context_window or self._default_n_ctx
        max_tokens = min(self.max_tokens, n_ctx)

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=Runner._n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=Runner._n_threads,
            max_tokens=max_tokens,
            verbose=VERBOSE_BACKEND,
        )

        if self.context_window is None:
            self.context_window = n_ctx
        try:
            metadata_ctx = getattr(self._llm, "metadata", {}).get("general.context_length")
            if isinstance(metadata_ctx, int):
                self.context_window = metadata_ctx
        except Exception:
            pass

        self.backend = Backend.LLAMACPP

    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str:
        if memory is None:
            memory = self.new_memory()
        memory.add_message(role, message)
        reply = self._llm.create_chat_completion(memory.get_history(self.backend))["choices"][0]["message"]["content"]
        memory.add_message(Role.ASSISTANT, reply)
        memory.debug_print()
        return reply

# ----------------------------------------------------------
# LangChainn
# ----------------------------------------------------------
class LangchainRunner(Runner):
    def __init__(self, context_window: Optional[int] = None) -> None:
        super().__init__(context_window=context_window)

    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str:
        if memory is None:
            memory = self.new_memory()
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
    #
##

class LangchainLocalRunner(LangchainRunner):
    def __init__(self, model: str, context_window: Optional[int] = None):
        """
        model_path:  local path of .GGUF file
        """
        super().__init__(context_window=context_window)
        self.llm = ChatOllama(model=model, verbose=VERBOSE_BACKEND)
        self.backend = Backend.OLLAMA

class LangchainRemoteRunner(LangchainRunner):
    def __init__(self, model: str | None = None, api_base: str | None = None, context_window: Optional[int] = None):
        """
        model:       e.g "gpt-4" or "gpt-3.5-turbo"
        api_base:    endpoint for OpenAI compatible API
        """
        super().__init__(context_window=context_window)
        #self._runner = ChatOpenAI(model=model, openai_api_base=api_base)
        self.backend = Backend.OTHER
        raise NotImplementedError


def _resolve_context_window(model: Model) -> Optional[int]:
    spec = model.value
    context_window = getattr(spec, "context_window", None)
    if isinstance(context_window, int) and context_window > 0:
        return context_window

    src = spec.source
    path: Optional[str] = None

    if isinstance(src, SourceFile):
        path = src.path
    elif isinstance(src, SourceLink):
        path = src.path
    elif isinstance(src, SourceHuggingface):
        path = src.local_dir + src.filename
    else:
        return None

    if not path or not Path(path).exists():
        return None

    try:
        reader = GGUFReader(path)
        field = reader.get_field("general.context_length")
        if isinstance(field, list) and field:
            field = field[0]
        if isinstance(field, (int, float)):
            return int(field)
    except Exception:
        return None

    return None
