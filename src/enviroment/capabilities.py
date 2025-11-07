from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, TYPE_CHECKING
from uuid import UUID

from enviroment.action import ActionTry, ActionType
from enviroment.exception import SoftException
from enviroment.world import World

if TYPE_CHECKING:
    from enviroment.entity import Entity
    from enviroment.interaction import Depth, ObserverPerception, PerceptionEnviroment


class Capability(ABC):
    """Composable behaviour unit that can be attached to an entity."""

    action_types: tuple[ActionType, ...] = ()

    def __init__(self, owner: Entity) -> None:
        self.owner = owner

    def supports(self, action_type: ActionType) -> bool:
        return action_type in self.action_types

    @abstractmethod
    def on_interact(self, actor_entity: Entity, action: ActionTry) -> str | None:
        ...

    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
        info: dict[str, object],
    ) -> None:
        """Allow the capability to enrich the perception output."""


class OpenableCapability(Capability):
    action_types = (ActionType.OPEN, ActionType.CLOSE)

    def __init__(
        self,
        owner: Entity,
        *,
        initially_open: bool = True,
        visibility_when_closed: float = 0.0,
    ) -> None:
        super().__init__(owner)
        self.is_open = initially_open
        self.visibility_when_closed = visibility_when_closed

    def on_interact(self, actor_entity: Entity, action: ActionTry) -> str | None:
        if action.type == ActionType.OPEN:
            if self.is_open:
                return "already open"
            lockable = self.owner.get_capability(LockableCapability)
            if lockable and lockable.is_locked:
                raise SoftException(
                    "The object is locked; unlock it before opening.",
                    console_message=(
                        f"Open blocked because '{self.owner.readable_id}' is locked."
                    ),
                    hint="Use the UNLOCK operator with the correct key first.",
                    context={
                        "target": self.owner.readable_id,
                        "action": ActionType.OPEN.value,
                    },
                )
            self.is_open = True
            return "opened"

        if action.type == ActionType.CLOSE:
            if not self.is_open:
                return "already closed"
            lockable = self.owner.get_capability(LockableCapability)
            if lockable and lockable.is_locked:
                raise SoftException(
                    "The object is locked and cannot be closed right now.",
                    console_message=(
                        f"Close blocked because '{self.owner.readable_id}' is locked."
                    ),
                    hint="Unlock it first or remove the locking mechanism.",
                    context={
                        "target": self.owner.readable_id,
                        "action": ActionType.CLOSE.value,
                    },
                )
            self.is_open = False
            return "closed"

        return None

    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
        info: dict[str, object],
    ) -> None:
        info["state"] = "open" if self.is_open else "closed"


class LockableCapability(Capability):
    action_types = (ActionType.UNLOCK, ActionType.LOCK)

    def __init__(
        self,
        owner: Entity,
        *,
        initially_locked: bool = False,
        allowed_keys: Iterable[UUID] | None = None,
    ) -> None:
        super().__init__(owner)
        self.is_locked = initially_locked
        self._allowed_keys: set[UUID] = set(allowed_keys or [])

    def allow_key(self, key_uuid: UUID) -> None:
        if key_uuid is not None:
            self._allowed_keys.add(key_uuid)

    def _has_key(self, action: ActionTry) -> bool:
        return action.item_1 in self._allowed_keys

    def on_interact(self, actor_entity: Entity, action: ActionTry) -> str | None:
        if action.type == ActionType.UNLOCK:
            if not self.is_locked:
                return "already unlocked"
            if not self._has_key(action):
                key_id = None
                if action.item_1:
                    key_entity = World.get_entity(action.item_1)
                    key_id = key_entity.readable_id if key_entity else str(action.item_1)
                raise SoftException(
                    "That key does not fit the lock.",
                    console_message=(
                        f"Unlock failed; key '{key_id or 'unknown'}' is not allowed on '{self.owner.readable_id}'."
                    ),
                    hint="Choose a key listed as compatible with this lock.",
                    context={
                        "target": self.owner.readable_id,
                        "key": key_id,
                        "action": ActionType.UNLOCK.value,
                    },
                )
            self.is_locked = False
            return "unlocked"

        if action.type == ActionType.LOCK:
            if self.is_locked:
                return "already locked"
            if not self._has_key(action):
                key_id = None
                if action.item_1:
                    key_entity = World.get_entity(action.item_1)
                    key_id = key_entity.readable_id if key_entity else str(action.item_1)
                raise SoftException(
                    "That key cannot lock this object.",
                    console_message=(
                        f"Lock failed; key '{key_id or 'unknown'}' is not allowed on '{self.owner.readable_id}'."
                    ),
                    hint="Use one of the keys referenced in the observation.",
                    context={
                        "target": self.owner.readable_id,
                        "key": key_id,
                        "action": ActionType.LOCK.value,
                    },
                )

            openable = self.owner.get_capability(OpenableCapability)
            if openable and openable.is_open:
                raise SoftException(
                    "Close the object before locking it.",
                    console_message=(
                        f"Lock prevented because '{self.owner.readable_id}' is still open."
                    ),
                    hint="Close it first, then lock it.",
                    context={
                        "target": self.owner.readable_id,
                        "action": ActionType.LOCK.value,
                    },
                )

            self.is_locked = True
            return "locked"

        return None

    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
        info: dict[str, object],
    ) -> None:
        info["lock_state"] = "locked" if self.is_locked else "unlocked"


class UsableCapability(Capability):
    action_types = (ActionType.USE,)

    def __init__(
        self,
        owner: Entity,
        effect: Callable[[Entity, ActionTry], str | None],
        *,
        perception_tag: str | None = None,
    ) -> None:
        super().__init__(owner)
        self._effect = effect
        self._perception_tag = perception_tag

    def on_interact(self, actor_entity: Entity, action: ActionTry) -> str | None:
        result = self._effect(actor_entity, action)
        return result if result is not None else "used"

    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
        info: dict[str, object],
    ) -> None:
        if self._perception_tag:
            info.setdefault("capabilities", []).append(self._perception_tag)
