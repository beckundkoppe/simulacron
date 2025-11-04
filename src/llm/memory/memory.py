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

    def to_string(self, backend: Backend) -> tuple[str, str, str]:
        return self.value, "", ""
        #if backend is Backend.LLAMACPP and self is Role.SYSTEM:
        #    return "user", "[system: ", "]"
        #else:
        #    return self.value, "", ""

class Memory(ABC):
    def __init__(self) -> None:
        self._history: List[Tuple[Role, str]] = []

    def __str__(self) -> str:
        return json.dumps(self.get_history(Backend.OTHER), ensure_ascii=False, indent=2)

    def add_message(self, role: Role, message: str) -> None:
        """Append a new message to the conversation history."""
        assert isinstance(role, Role), "role must be a Role enum"
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        self._history.append((role, message))

    @abstractmethod
    def summarize(self, max_tokens: Optional[int] = None) -> str:
        """
        Optional: return a concise summary of the conversation.
        max_tokens: hint for the desired size of the summary.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [(role.name, msg) for role, msg in self._history],
                f,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path: str) -> None:
        """Load the memory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._history = [(Role[name], msg) for name, msg in data]
    
    def get_history(self, backend: Backend) -> List[dict[str, str]]:
        """
        Export the history in the format required by the target backend,
        preserving all ANSI colors or markdown.
        """
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
        from config import pretty, bullet, debug_separator, Color

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
