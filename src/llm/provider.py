from abc import ABC, abstractmethod
import copy
from datetime import datetime
import json
from pathlib import Path
import re
from typing import List, Optional, Tuple
from langchain_openai import ChatOpenAI

import config
import debug
from llm.cache import Cache
from llm.memory.memory import Memory, Role, Type
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

    def call(self, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None) -> str:
        return self.invoke(message=None, transient=transient, role=role, override=override, append=False)

    def _invoke_pre(self, message: Optional[str] = None, transient: Optional[str] = None, role: Role = Role.USER, override: Optional[Memory] = None, append: bool = True) -> List[dict[str, str]]:
        if message is None:
            msg = ""
        else:
            msg = message

        assert isinstance(msg, str), "message ist no string"

        console.pretty(console.bullet(f"[{role.to_string()}] {msg + (" " + transient if transient is not None else "")}", color=console.Color.CYAN))

        if override is not None:
            mem = override
        else:
            mem = self.memory

        if append and message:
            mem.append_message(role, message)


        out = mem.get_history()

        if transient is not None:
            out.append({
                "role": role.to_string(),
                "content": transient
            })

        self._log_raw_event(
            "request",
            {"messages": out},
            attempt=1,
        )

        return out

    def _clean_reply(reply: str) -> str:
        if isinstance(reply, str):
            return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        else:
            return ""
        
    def _hard_clean_reply(reply: str) -> str:
        if "<think>" in reply and "</think>" not in reply:
            return ""
        return reply
    def _invoke_post(self, reply: str, override: Optional[Memory] = None, append: bool = True) -> None:
        assert isinstance(reply, str)

        console.pretty(console.bullet(f"[{self.name}] {reply}", color=console.Color.YELLOW))

        if override is not None:
            if append and len(reply) > 0:
                override.append_message(Role.ASSISTANT, reply)
        else:
            if not self.has_memory:
                self.memory = Memory()
            else:
                if append and len(reply) > 0:
                    self.memory.append_message(Role.ASSISTANT, reply)

    def _log_raw_event(self, event: str, payload: dict, attempt: Optional[int] = None) -> None:
        if config.APPEND_RAW is None:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "provider": self.name,
            "model": getattr(self.model.value, "name", str(self.model)),
            "event": event,
        }

        if attempt is not None:
            entry["attempt"] = attempt

        entry.update(payload)

        try:
            config.APPEND_RAW(json.dumps(entry, default=str) + "\n")
        except Exception:
            # Logging failures should not break inference.
            pass

    def build(name: str, model: Model, memory: Optional[Memory] = None) -> "Provider":
        src = model.value.source

        if isinstance(src, SourceOllama):
            return LangchainLocalProvider(name, model=model,memory=memory)
        elif isinstance(src, SourceRemote):
            return LangchainRemoteProvider(name, model=model,memory=memory)
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

        if debug.VERBOSE_LLAMACPP:
            print(temp)

        #response_format={"type": "text"}
        reply = self.llm.create_chat_completion(temp)["choices"][0]["message"]["content"]
        self._log_raw_event(
            "response",
            {
                "reply": reply,
                "clean_reply": Provider._clean_reply(reply),
            },
            attempt=1,
        )

        clean_reply = Provider._hard_clean_reply(reply)

        if debug.VERBOSE_LLAMACPP:
            print(clean_reply)

        if len(clean_reply) <= 0:
            hist = copy.deepcopy(temp)
            hist[-1]["content"] = hist[-1]["content"] + " /nothink"
            self._log_raw_event(
                "request",
                {"messages": hist, "note": "retry_without_think"},
                attempt=2,
            )
            reply = self.llm.create_chat_completion(hist)["choices"][0]["message"]["content"]
            self._log_raw_event(
                "response",
                {
                    "reply": reply,
                    "clean_reply": Provider._clean_reply(reply),
                },
                attempt=2,
            )

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

        if debug.VERBOSE_LANGCHAIN:
            print(temp)

        reply = self.llm.invoke(temp).content
        self._log_raw_event(
            "response",
            {
                "reply": reply,
                "clean_reply": Provider._clean_reply(reply),
            },
            attempt=1,
        )

        if debug.VERBOSE_LANGCHAIN:
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
        src = model.value.source

        self.llm = ChatOpenAI(

            model=src.model_id,
            base_url=src.endpoint_url,
            api_key="none"
        )
