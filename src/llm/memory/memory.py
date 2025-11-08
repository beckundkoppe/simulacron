from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Optional, TYPE_CHECKING
import json

from llm.model import Backend

if TYPE_CHECKING:
    from llm.cache import Cache
    from llm.model import Model
    from llm.runner import Runner


def _approximate_token_count(text: str) -> int:
    """Return a rough token count for a text snippet."""
    # Using whitespace splitting provides a cheap lower bound that does not
    # require a backend specific tokenizer. We still ensure a minimum of one
    # token so that empty or whitespace-only messages do not break the
    # heuristics that depend on the value.
    tokens = len(text.split())
    return max(1, tokens)

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

    def copy(self) -> "Memory":
        """
        Return a deep copy of the memory, including its history.
        """
        import copy
        new_copy = self.__class__()
        new_copy._history = copy.deepcopy(self._history)
        return new_copy

    def __str__(self) -> str:
        return json.dumps(self.get_history(Backend.OTHER), ensure_ascii=False, indent=2)

    def add_message(self, role: Role, message: str) -> None:
        """Append a new message to the conversation history."""
        assert isinstance(role, Role), "role must be a Role enum"
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        self._history.append((role, message))

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

    def debug_print_recent(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        """
        Pretty-print the two most recent conversation messages with colorised bullets.
        """
        from debug.console import pretty, bullet, debug_separator, Color

        color = Color.RED if is_agent else Color.BLUE
        lines: list[str] = [debug_separator(color=color)]

        # take only the two most recent messages
        for role, msg in self._history[-2:]:
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


class _SilentMemory(Memory):
    """Memory variant that suppresses debug output for internal prompts."""

    def debug_print(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        return

    def debug_print_recent(self, is_agent: bool = False, name: Optional[str] = None) -> None:
        return


class SummarizingMemory(Memory):
    """Memory that compresses older messages when the context nears exhaustion."""

    _SUMMARY_PREFIX = "[conversation-summary]"

    def __init__(
        self,
        max_tokens: int = 4096,
        trigger_ratio: float = 0.9,
        preserve_recent: int = 6,
        *,
        cache: Optional["Cache"] = None,
        model: Optional["Model"] = None,
        summary_instructions: Optional[str] = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        super().__init__()
        self.max_tokens = int(max_tokens)
        self.trigger_ratio = trigger_ratio
        self.preserve_recent = max(2, preserve_recent)
        if cache is not None and model is None:
            raise ValueError("model must be provided when cache is configured")

        self._cache = cache
        self._model = model
        self._summary_runner: Optional["Runner"] = None
        self._summary_instructions = summary_instructions or (
            "You compress conversations into short, factual summaries. "
            "Keep goals, decisions, and unresolved questions. "
            "Use at most {limit} words."
        )

    def copy(self) -> "SummarizingMemory":
        import copy

        new_copy = self.__class__(
            self.max_tokens,
            trigger_ratio=self.trigger_ratio,
            preserve_recent=self.preserve_recent,
            cache=self._cache,
            model=self._model,
            summary_instructions=self._summary_instructions,
        )
        new_copy._history = copy.deepcopy(self._history)
        new_copy._summary_runner = self._summary_runner
        return new_copy

    def add_message(self, role: Role, message: str) -> None:
        super().add_message(role, message)
        self._ensure_within_context()

    def _ensure_within_context(self) -> None:
        if not self.max_tokens:
            return

        current_tokens = sum(_approximate_token_count(msg) for _, msg in self._history)
        threshold = int(self.max_tokens * self.trigger_ratio)
        if current_tokens < threshold:
            return

        self._summarize_history()

    def _summarize_history(self) -> None:
        if len(self._history) <= self.preserve_recent:
            return

        head = self._history[:-self.preserve_recent]
        tail = self._history[-self.preserve_recent:]

        preserved_system: List[Tuple[Role, str]] = []
        summarizable: List[Tuple[Role, str]] = []

        for role, msg in head:
            if role is Role.SYSTEM and msg.startswith(self._SUMMARY_PREFIX):
                summarizable.append((role, msg[len(self._SUMMARY_PREFIX):].lstrip()))
            elif role is Role.SYSTEM:
                preserved_system.append((role, msg))
            else:
                summarizable.append((role, msg))

        if not summarizable:
            return

        summary_text = self._build_summary(summarizable)
        combined_summary = f"{self._SUMMARY_PREFIX} {summary_text}"

        # Merge with existing summary if one exists in preserved_system
        for i, (role, msg) in enumerate(preserved_system):
            if role is Role.SYSTEM and msg.startswith(self._SUMMARY_PREFIX):
                old_summary = msg[len(self._SUMMARY_PREFIX):].lstrip()
                merged = f"{self._SUMMARY_PREFIX} {old_summary} {summary_text}"
                preserved_system[i] = (Role.SYSTEM, merged)
                break
        else:
            preserved_system.append((Role.SYSTEM, combined_summary))

        self._history = preserved_system + tail


    def _build_summary(self, messages: List[Tuple[Role, str]]) -> str:
        if not messages:
            return ""

        runner = self._get_runner()
        summary_limit = max(32, self.max_tokens // 8)

        if runner is not None:
            try:
                return self._summarize_with_llm(messages, runner, summary_limit)
            except Exception:
                pass

        return self._fallback_summary(messages, summary_limit)

    def _get_runner(self) -> Optional["Runner"]:
        if self._cache is None or self._model is None:
            return None
        if self._summary_runner is None:
            self._summary_runner = self._cache.get(self._model)
        return self._summary_runner

    def _summarize_with_llm(
        self,
        messages: List[Tuple[Role, str]],
        runner: "Runner",
        limit: int,
    ) -> str:
        formatted = self._format_messages(messages)
        memory = _SilentMemory()
        instructions = self._summary_instructions.format(limit=limit)
        memory.add_message(Role.SYSTEM, instructions)
        summary = runner.invoke(formatted, memory=memory)
        return summary.strip()

    def _fallback_summary(self, messages: List[Tuple[Role, str]], limit: int) -> str:
        parts: List[str] = []
        for role, msg in messages:
            clean_msg = " ".join(str(msg).split())
            if not clean_msg:
                continue
            parts.append(f"{role.value}: {clean_msg}")

        if not parts:
            return ""

        words = " ".join(parts).split()
        if len(words) <= limit:
            return " ".join(words)

        truncated = " ".join(words[:limit]) + " ..."
        return truncated

    def _format_messages(self, messages: List[Tuple[Role, str]]) -> str:
        lines: List[str] = []
        for role, msg in messages:
            clean_msg = " ".join(str(msg).split())
            lines.append(f"{role.value.upper()}: {clean_msg}")
        return "\n".join(lines)
