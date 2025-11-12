from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Optional

import debug
from llm.cache import Cache
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
from util import console

# ----------------------------------------------------------
# Baseclass
# ----------------------------------------------------------
@abstractmethod
class Provider(ABC):
    def _init(self, name: str, model: Model, memory: Optional[Memory] = None) -> None:
        self.name = name
        self.model = model
        
        if memory is None:
            self.has_memory = False
            self.memory = Memory()
        else:
            self.has_memory = True
            self.memory = memory

    @abstractmethod
    def invoke(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str: ...

    def _invoke_pre(self, message: str, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> Memory:
        assert isinstance(message, str), "message ist no string"

        console.pretty(console.bullet(f"[{role.to_string()}] {message + (" " + transient if transient is not None else "")}", color=console.Color.CYAN))

        if override is not None:
            mem = override
        else:
            mem = self.memory

        if append:
            mem.add_message(role, message)

        temp = mem.copy()
        if transient is not None:
            temp.add_message(role, transient)

        console.pretty()

        return temp

    def _clean_reply(reply: str) -> str:
        if isinstance(reply, str):
            return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        else:
            return ""

    def _invoke_post(self, reply: str, override: Optional[Memory] = None, append: bool = True) -> None:
        assert isinstance(reply, str)

        console.pretty(console.bullet(f"[{self.name}] {reply}", color=console.Color.YELLOW))

        if override is not None:
            if append and len(reply) > 0:
                override.add_message(Role.ASSISTANT, reply)
        else:
            if not self.has_memory:
                self.memory = Memory()
            else:
                if append and len(reply) > 0:
                    self.memory.add_message(Role.ASSISTANT, reply)

    def build(name: str, model: Model, memory: Optional[Memory] = None) -> "Provider":
        src = model.value.source

        if isinstance(src, SourceOllama):
            return LangchainLocalProvider(name, model=model,memory=memory)
        elif isinstance(src, SourceRemote):
            raise NotImplementedError
        elif isinstance(src, SourceFile) or isinstance(src, SourceLink) or isinstance(src, SourceHuggingface):
            return LlamaCppProvider(name, model=model, memory=memory)
        else:
            raise ValueError("Unsupported Provider configuration")

# ----------------------------------------------------------
# LlamaCppProvider
# ----------------------------------------------------------
class LlamaCppProvider(Provider):
    def __init__(self, name: str, model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)
        self._init(name, model, memory)

    def _init(self, name: str, model: Model, memory: Optional[Memory] = None):
        self.llm = Cache.get(model)

    def invoke(self, message: str, transient: Optional[str] = None,  role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

        reply = self.llm.create_chat_completion(temp.get_history())["choices"][0]["message"]["content"]

        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)

        return clean_reply

# ----------------------------------------------------------
# LangChainProvider
# ----------------------------------------------------------
class LangchainProvider(Provider):
    def _init(self, name: str, model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)

    def invoke(self, message: str, transient: Optional[str] = None,  role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> str:
        temp = self._invoke_pre(message=message, transient=transient, role=role, override=override, append=append)

        if debug.VERBOSE_OLLAMA:
            print(message)

        reply = self.llm.invoke(temp.get_history()).content

        if debug.VERBOSE_OLLAMA:
            print(reply)
        
        clean_reply = Provider._clean_reply(reply)
        self._invoke_post(reply=clean_reply, override=override, append=append)
        return clean_reply

class LangchainLocalProvider(LangchainProvider):
    def __init__(self, name: str, model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)
        self._init(name, model, memory)

    def _init(self, name: str, model: Model, memory: Optional[Memory] = None):
        self.llm = Cache.get(model)

class LangchainRemoteProvider(LangchainProvider):
    def __init__(self, name: str, model: Model, memory: Optional[Memory] = None):
        super()._init(name, model, memory)
        self._init(name, model, memory)

    def _init(self, name: str, model: Model, memory: Optional[Memory] = None):
        #self._runner = ChatOpenAI(model=model, openai_api_base=api_base)
        raise NotImplementedError