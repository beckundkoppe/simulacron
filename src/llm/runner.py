from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional
from llama_cpp import Llama
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

from debug.settings import VERBOSE_BACKEND
from llm.memory.memory import Memory, Role
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
            verbose=VERBOSE_BACKEND,
        )

        self.backend = Backend.LLAMACPP
    
    def invoke(self, message: str, role: Role = Role.USER, memory: Optional[Memory] = None) -> str:
        if memory is None:
            memory = Memory()
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
            memory = Memory()
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
    def __init__(self, model: str):
        """
        model_path:  local path of .GGUF file
        """ 
        self.llm = ChatOllama(model=model, verbose=VERBOSE_BACKEND)
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
