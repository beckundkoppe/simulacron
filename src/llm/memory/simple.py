import json
from typing import List, Tuple, Optional

from llm.memory.memory import Memory, Role
from llm.model import Backend

class SimpleMemory(Memory):
    """
    Conversation memory that stores the full history permanently.
    No summarization or token-based pruning.
    """

    def __init__(self) -> None:
        super().__init__()

    def add_message(self, role: Role, message: str) -> None:
        """Append a new message to the conversation history."""
        assert isinstance(role, Role), "role must be a Role enum"
        if not isinstance(message, str):
            raise TypeError(f"Memory expects plain text messages, got {type(message).__name__}")
        self._history.append((role, message))

    def summarize(self, max_tokens: Optional[int] = None) -> str:
        """
        Return the conversation as plain text.
        If max_tokens is given, keep whole messages in order,
        stop once adding the next message would exceed the limit.
        """
        if max_tokens is None:
            return "\n".join(f"{role.name}: {msg}" for role, msg in self._history)

        collected: list[str] = []
        token_count = 0

        for role, msg in self._history:
            text = f"{role.name}: {msg}"
            
            # simple whitespace tokens
            tokens = text.split()
            
            needed = len(tokens)

            # check if adding this message would exceed the limit
            if token_count + needed > max_tokens:
                break

            collected.append(text)
            token_count += needed

        return "\n".join(collected)

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

    def get_history(self, backend: Backend) -> list[dict[str, str]]:
        """
        Export the history in the format required by the target backend,
        preserving all ANSI colors or markdown.
        """
        messages: list[dict[str, str]] = []
        for role, msg in self._history:
            role_str, front, end = role.to_string(backend)
            messages.append({
                "role": role_str,
                "content": front + msg + end
            })
        return messages
