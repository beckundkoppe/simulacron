from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Optional

from llama_cpp import Llama
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama

import debug
from llm.memory.memory import Memory, Role
from llm.prepare import prepare_model_source
from llm.model import (
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
@abstractmethod
class Provider(ABC):
    _n_gpu_layers = -1
    _n_threads = 24

    def _init(self, model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning = reasoning
        self.seed = seed
        
        if memory is None:
            self.has_memory = False
            self.memory = Memory()
        else:
            self.has_memory = True
            self.memory = memory

    @abstractmethod
    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str: ...

    def _invoke_pre(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None) -> Memory:
        assert isinstance(message, str)

        if override is not None:
            mem = override
        else:
            mem = self.memory.add_message(role, message)

        temp = mem.copy()
        temp.add_message(role, transient)

        return temp

    def _clean_reply(reply: str) -> str:
        if isinstance(reply, str):
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        else:
            return ""

    def _invoke_post(self, reply: str, override: Optional[Memory] = None, append = True) -> None:
        assert isinstance(reply, str)

        if override is not None:
            if append and len(reply) > 0:
                override.add_message(Role.ASSISTANT, reply)
        else:
            if not self.has_memory:
                self.memory = Memory()
            else:
                if append and len(reply) > 0:
                    self.memory.add_message(Role.ASSISTANT, reply)

    def build(model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None) -> "Provider":
        prepare_model_source(model)
        src = model.value.source

        if isinstance(src, SourceOllama):
            return LangchainLocalProvider(model=model, temperature=temperature, reasoning=reasoning, seed=seed, memory=memory)
        elif isinstance(src, SourceRemote):
            raise NotImplementedError
        elif isinstance(src, SourceFile) or isinstance(src, SourceLink) or isinstance(src, SourceHuggingface):
            return LlamaCppProvider(model=model, temperature=temperature, reasoning=reasoning, seed=seed, memory=memory)
        else:
            raise ValueError("Unsupported Provider configuration")

# ----------------------------------------------------------
# LlamaCppProvider
# ----------------------------------------------------------
class LlamaCppProvider(Provider):
    def __init__(self, model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None):
        super()._init(model, temperature, reasoning, seed, memory)

        src = model.value.source

        if isinstance(src, SourceFile):
            path = src.path
        elif isinstance(src, SourceLink):
            path = src.path
        elif isinstance(src, SourceHuggingface):
            path = src.local_dir + src.filename
        else:
            raise ValueError("Unsupported LlamaCppProvider Source")

        if not Path(path).exists():
            raise FileNotFoundError(f"Cant load model: file not found: {path}")

        self.llm = Llama(
            model_path=path,
            n_gpu_layers=Provider._n_gpu_layers,
            n_ctx=0,
            n_threads=Provider._n_threads,
            verbose=debug.VERBOSE_LLAMACPP,
        )

    def invoke(self, message: str, transient: Optional[str] = None,  role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override)

        reply = self.llm.create_chat_completion(temp.get_history())["choices"][0]["message"]["content"]

        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

# ----------------------------------------------------------
# LangChainProvider
# ----------------------------------------------------------
class LangchainProvider(Provider):
    def _init(self, model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None):
        super()._init(model, temperature, reasoning, seed, memory)

    def invoke(self, message: str, transient: Optional[str] = None,  role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override)

        reply = self.llm.invoke(temp.get_history()).content
        
        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

class LangchainLocalProvider(LangchainProvider):
    def __init__(self, model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None):
        super()._init(model, temperature, reasoning, seed, memory)

        self.llm = ChatOllama(model=model, verbose=debug.VERBOSE_OLLAMA)

class LangchainRemoteProvider(LangchainProvider):
    def __init__(self, model: Model, temperature: float, reasoning: bool = True, seed =12345, memory: Optional[Memory] = None):
        super()._init(model, temperature, reasoning, seed, memory)

        #self._runner = ChatOpenAI(model=model, openai_api_base=api_base)
        raise NotImplementedError