from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Optional
import json

from llm.model import Backend

class MemoryType(str, Enum):
    SIMPLE = "simple"
    WINDOW = "window"
    SUMMARY = "summary"
    VECTOR = "vector"

class Role(Enum):
    SYSTEM      = "system"
    USER        = "user"
    ASSISTANT   = "assistant"
    TOOL        = "tool"

    def to_string(self, backend: Backend) -> tuple[str, str, str]:
        return self.value, "", ""

class Memory(ABC):
    def __init__(self) -> None:
        self._history: List[Tuple[Role, str]] = []

    def __str__(self) -> str:
        return json.dumps(self.get_history(Backend.OTHER), ensure_ascii=False, indent=2)

    @abstractmethod
    def add_message(self, role: Role, message: str) -> None:
        """
        Add a new message to the memory.
        role: typically "user" or "assistant"
        message: plain text (could contain markdown or ANSI)
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self, max_tokens: Optional[int] = None) -> str:
        """
        Optional: return a concise summary of the conversation.
        max_tokens: hint for the desired size of the summary.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the memory to disk or any other storage backend.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the memory from disk or any other storage backend.
        """
        raise NotImplementedError
    
    def get_history(self, backend: Backend) -> List[dict[str, str]]:
        messages: List[dict[str, str]] = []
        for role, msg in self._history:
            role_str, front, end = role.to_string(backend)

            if not isinstance(msg, str):
                msg = str(msg)

            messages.append({
                "role": role_str,
                "content": front + msg + end
            })
        return messages
    
    def debug_print(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        """
        Pretty-print the current conversation history with colorised bullets.
        """
        from debug.console import pretty, bullet, debug_separator, Color

        color = Color.BLUE

        if is_agent: color = Color.RED

        lines: list[str] = [debug_separator(color=color)]

        for role, msg in self._history:
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
