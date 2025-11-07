from __future__ import annotations

from typing import Any, Dict, List


class Resultbuffer:
    buffer: List["Result"] = []


class Result:
    def __init__(
        self,
        agent_message: str,
        console_message: str | None = None,
        *,
        hint: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        self.agent_message = agent_message
        self.console_message = console_message or agent_message
        self.hint = hint
        self.context: Dict[str, Any] = dict(context or {})
        Resultbuffer.buffer.append(self)

    # Backwards compatibility for legacy access patterns.
    @property
    def what(self) -> str:
        return self.agent_message


class FormalError(Result):
    def __init__(
        self,
        agent_message: str,
        console_message: str | None = None,
        *,
        hint: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_message, console_message, hint=hint, context=context)


class ActionNotPossible(Result):
    def __init__(
        self,
        agent_message: str,
        console_message: str | None = None,
        *,
        hint: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_message, console_message, hint=hint, context=context)


class Success(Result):
    def __init__(
        self,
        agent_message: str,
        console_message: str | None = None,
        *,
        hint: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_message, console_message, hint=hint, context=context)
