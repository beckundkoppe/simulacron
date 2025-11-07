from __future__ import annotations

from typing import Any, Mapping


class ActionException(Exception):
    """Base exception carrying rich feedback for agent and console output."""

    def __init__(
        self,
        agent_message: str,
        console_message: str | None = None,
        *,
        hint: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_message)
        self.agent_message = agent_message
        self.console_message = console_message or agent_message
        self.hint = hint
        self.context = dict(context or {})


class HardException(ActionException):
    pass


class SoftException(ActionException):
    pass


class SuccessException(Exception):
    pass
