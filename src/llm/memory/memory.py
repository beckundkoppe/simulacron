from abc import ABC, abstractmethod
from enum import Enum, auto
import re
from typing import Iterable, List, Tuple, Optional, TYPE_CHECKING
import json

import config
import current
from llm.model import Backend

if TYPE_CHECKING:
    from llm.cache import Cache
    from llm.model import Model

class Role(Enum):
    SYSTEM      = "system"
    USER        = "user"
    ASSISTANT   = "assistant"
    TOOL        = "tool"

    def to_string(self) -> str:
        return self.value

class Type(Enum):
    GOAL        = auto()
    FEEDBACK    = auto()
    OBSERVATION = auto()
    PERCEPTION = auto()
    CURRENT_OBSERVATION = auto()
    REFLECT     = auto()
    PLAN        = auto()
    SUMMARY     = auto()

class Memory(ABC):
    def __init__(self, goal = None, path = None) -> None:
        self._goal: str = goal
        self.path = path
        self._history: List[Tuple[Type, Role, str]] = []

    def copy(self) -> "Memory":
        """
        Return a deep copy of the memory, including its history.
        """
        import copy
        new_copy = self.__class__(self._goal, self.path)
        new_copy._history = copy.deepcopy(self._history)
        return new_copy

    def __str__(self) -> str:
        return json.dumps(self._get_history(Backend.OTHER), ensure_ascii=False, indent=2)
    
    def append(self, other: List[Tuple[Type, Role, str]]) -> None:
        self._history.extend(other._history)

    def get_last_n(self, n: int):
        return self._history[:-n]

    def append_message(self, role: Role, message: str, type: Optional[Type] = None) -> None:
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        self._history.append((type, role, message))

    def prepend_message(self, role: Role, message: str, type: Optional[Type] = None) -> None:
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        
        self._history.insert(0, (type, role, message))

    def save(self):
        if(self.path): self._store(self.path)

    def _store(self, path: str) -> None:
        self._save(path, self._history)

    def _save(self, path: str, history) -> None:
        """Persist the memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [(type.name if type is not None else "NONE", role.name, msg) for type, role, msg in history],
                f,
                ensure_ascii=False,
                indent=2
            )

    def _load(self, path: str):
        """Load the memory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(Role[name], msg) for name, msg in data]

    def get_history(self) -> List[dict[str, str]]:
        return self._build_history(self._history)
    
    def _build_history(self, history) -> List[dict[str, str]]:
        messages: List[dict[str, str]] = []
        for type, role, msg in history:
            role_str = role.to_string()

            if not isinstance(msg, str):
                msg = str(msg)

            messages.append({
                "role": role_str,
                "content": msg
            })
        return messages
    
    def debug_print(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        """
        Pretty-print the current conversation history with colorised bullets.
        """
        from util.console import pretty, bullet, debug_separator, Color

        color = Color.BLUE

        if is_agent: color = Color.RED

        lines: list[str] = [debug_separator(color=color)]

        for type, role, msg in self._history:
            if role is Role.USER:
                lines.append(
                    bullet(f"{msg!s}", bullet="[user]      ", color=Color.CYAN)
                )
            elif role is Role.ASSISTANT:
                lines.append(
                    bullet(f"{msg!s}", bullet="[assistant] ", color=Color.YELLOW)
                )
            elif role is Role.SYSTEM:
                lines.append(
                    bullet(f"{msg!s}", bullet="[system]    ", color=Color.MAGENTA)
                )
            else:
                lines.append(
                    bullet(f"{msg!s}", bullet=f"[{role.name.lower()}] ", color=Color.WHITE)
                )

        lines.append(debug_separator(color=color))

        pretty(*lines)

    def debug_print_recent(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        """
        Pretty-print the two most recent conversation messages with colorised bullets.
        """
        from util.console import pretty, bullet, debug_separator, Color

        color = Color.RED if is_agent else Color.BLUE
        lines: list[str] = [debug_separator(color=color)]

        # take only the two most recent messages
        for type, role, msg in self._history[-2:]:
            if role is Role.USER:
                lines.append(
                    bullet(f"{msg!s}", bullet="[user]      ", color=Color.CYAN)
                )
            elif role is Role.ASSISTANT:
                lines.append(
                    bullet(f"{msg!s}", bullet="[assistant] ", color=Color.YELLOW)
                )
            elif role is Role.SYSTEM:
                lines.append(
                    bullet(f"{msg!s}", bullet="[system]    ", color=Color.MAGENTA)
                )
            else:
                lines.append(
                    bullet(f"{msg!s}", bullet=f"[{role.name.lower()}] ", color=Color.WHITE)
                )

        lines.append(debug_separator(color=color))
        pretty(*lines)

    def get_token_count(self) -> int:
        return Memory._approximate_token_count(self._history)

    def _approximate_token_count(text: str) -> int:
        """Heuristic token estimator that roughly matches LLM tokenization behavior."""
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, len(tokens))
    
    def assure_max_token_count(self, max_count: int):
        size = Memory._approximate_token_count(str(self._history))

        while(size > max_count):
            todo = min(max(size - max_count, (size - max_count) * 10), int(config.Backend.n_context / 4))

            staged = []

            his = self._history.copy()

            for x in his:
                type, role, msg = x
                todo -= Memory._approximate_token_count(role.to_string() + msg)
                staged.append(x)

                if(todo < 0) and len(staged) > 1:
                    break

            if len(staged) <= 1:
                raise Exception("context to small")

            for s in staged:
                self._history.remove(s)

            print(f"staged token for summary: {Memory._approximate_token_count(str(staged))}")
            type, role, summary = self.summarize(staged)

            print(f"token after summary: {Memory._approximate_token_count(str((type, role, summary)))}")
                        
            self.prepend_message(role, summary, type)
            
            print(f"new memory token size: {Memory._approximate_token_count(str(self._history))}")

            size = Memory._approximate_token_count(str(self._history))

    def summarize(self, messages: List[Tuple[Type, Role, str]]) -> Tuple[Type, Role, str]:
        from llm.provider import Provider

        data = str(messages)

        mem = Memory()
        mem.append_message(Role.USER, data)
        summarizer = Provider.build("summarizer", current.EXTRA_MODEL, memory=mem)
        summary = summarizer.call("Summarize only the valuable parts in few short sentences. (no 'summary:' or similar).")
        
        return (Type.SUMMARY, Role.USER, summary)
    
    def append_message(self, role: Role, message: str, type: Optional[Type] = None) -> None:
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        self._history.append((type, role, message))
    
    def set_plan(self, plan):
        self.append_message(Role.SYSTEM, plan, Type.PLAN)
