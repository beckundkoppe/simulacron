
from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING, Type, TypeVar

from enviroment.action import ActionTry, ActionType
from enviroment.capabilities import LockableCapability, OpenableCapability
from enviroment.exception import HardException, SoftException, SuccessException
import config as config
from enviroment.room import Position, Room
from enviroment.world import World
from enviroment.interaction import CompositeDatum, Datum, DatumOperator, Depth, Interaction, PerceptionEnviroment, ObserverPerception, SensoryChannel

if TYPE_CHECKING:
    from enviroment.capabilities import Capability

T = TypeVar("T", bound="Capability")

class Entity:
    def __init__(self, name: str, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0, is_collectible: bool = False) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.pos = pos

        self.material = material
        self.description = description
        self.uniqueness = uniqueness
        self.prominence = prominence
        self.is_collectible = is_collectible

        self.room: Room | None = None

        self.entity_id: int | None = None
        self.readable_id: str | None = None

        self._capabilities: list[Capability] = []

        World.add_entity(self)

    # -- capability plumbing -------------------------------------------------
    def add_capability(self, capability: Capability) -> None:
        if capability.owner is not self:
            raise ValueError("capability owner mismatch")
        self._capabilities.append(capability)

    def iter_capabilities(self) -> Iterable[Capability]:
        return tuple(self._capabilities)

    def get_capability(self, capability_type: Type[T]) -> T | None:
        for capability in self._capabilities:
            if isinstance(capability, capability_type):
                return capability
        return None

    def _ensure_same_room(self, actor_entity: "Entity") -> None:
        if self.room is None:
            raise HardException("entity is not registered in the world")

        actor_room = actor_entity.room
        if actor_room is None or not actor_room.contains_entity(self):
            raise HardException(f"{self.readable_id} is not in your room")

    def enter(self, room: "Room") -> None:
        if self.room is not None:
            print(f"can not enter room {room}: {self.readable_id} is already in a room")
            return

        if room.room_id is None:
            raise ValueError("Room is not registered in the world")

        if self.pos is not None:
            assert room.isPosInRoom(self.pos), (
                f"Position: ({self.pos.x, self.pos.y}) from {self.name} does not fit in Room:"
                f"{room.name} ({room.extend_x,room.extend_y})"
            )

        assert self.entity_id is not None, "entity must be registered before entering a room"
        room.entities.add(self)
        self.room = room

    def leave(self):
        if self.room is not None:
            self.room.entities.discard(self)
            self.room = None
        else:
            print(f"can not leave room: {self.readable_id} is in no room")

    def hasChild(self, entity: "Entity") -> bool:
        return False

    def on_interact(
        self,
        actor_entity,
        action: ActionTry
    ) -> str:
        self._ensure_same_room(actor_entity)

        for capability in self._capabilities:
            if capability.supports(action.type):
                result = capability.on_interact(actor_entity, action)
                if result is not None:
                    return result

        raise SoftException("had no effekt")

    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:
    
        self.info: dict[str, object] = {}

        name = Datum(
            self,
            key="name",
            value=self.name,
            channel_intensity={
                SensoryChannel.VISION: 0.5 # very easy to spot once you look
            },
            min_depth=Depth.MINIMAL,
            detect_threshold=0.1,
        )

        material = Datum(
            self,
            key="material",
            value=self.material,
            channel_intensity={
                SensoryChannel.VISION: 0.5, # need a clear look at the grain
                SensoryChannel.TOUCH:  0.3 # need to run your hand over it
            },
            min_depth=Depth.REDUCED,
            detect_threshold = 0.3,
            op=DatumOperator.AND,
        )

        description = Datum(
            self,
            key="description",
            value=self.description,
            channel_intensity={
                SensoryChannel.VISION: 0.3 # requires a normal careful look
            },
            min_depth=Depth.NORMAL,
            detect_threshold = 0.3,
        )

        condition = lambda: self._can_identify_uniquely(observer, env)
        id = CompositeDatum(self, "id", self.readable_id, condition, Depth.EXTENDED)

        #{states & effects would go here}

        datums: list[Datum] = [name]
        if (self.material): datums.append(material)
        if (self.description): datums.append(description)
        datums.append(id)

        for d in datums:
            d.perceive(observer, env, depth)

        self.is_any_perceived = self.info.get("name") is not None

        if(self.is_any_perceived and not condition()): 
            self.info["id"] = "not uniquely identifiable at a glance"

        if self.room is not None and self.pos is not None:
            self.info["position"] = self.pos.map(self.room).toString()

        for capability in self._capabilities:
            capability.on_perceive(observer, env, depth, self.info)

        return self.info
    
    def _can_identify_uniquely(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
    ) -> bool:
        """
        True if the observer can uniquely identify the entity:
        requires strong vision AND (hearing OR touch).

        Uses PerceptionContext.effort_for_channel() to scale each
        sensory score by the current interaction effort.

        The `uniqueness` factor (0..1) makes the test easier or harder:
            0.0 = very generic object (hard),
            1.0 = highly distinctive (easy).
        """

        if config.CONFIG is not config.PerceptionType.SENSE:
            return True

        for action in env.interactions:
            v = min(
                1.0,
                observer.vision_effective(env, intensity=1.0)
                * action.channel_effort(SensoryChannel.VISION) * self.prominence
            )
            h = min(
                1.0,
                observer.hear_sound_effective(env)
                * action.channel_effort(SensoryChannel.HEARING) * self.prominence
            )
            t = min(
                1.0,
                observer.touch_effective(env, intensity=1.0)
                * action.channel_effort(SensoryChannel.TOUCH) * self.prominence
            )

            # scale thresholds inversely with uniqueness:
            # high uniqueness → lower threshold to pass
            vision_req  = 0.6 * (1.0 - 0.5 * self.uniqueness)
            second_req  = 0.4 * (1.0 - 0.5 * self.uniqueness)

            if v >= vision_req and (h >= second_req or t >= 0.3 * (1.0 - 0.5 * self.uniqueness)): return True

        return False
    
class ContainerEntity(Entity):
    def __init__(self, name: str, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0, is_collectible: bool = False, visibility: float = 0.8) -> None:
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible)

        self.visibility = visibility
        self.children: List[Entity] = []

    def _would_form_cycle(self, candidate: "ContainerEntity") -> bool:
        stack = [candidate]
        visited: set[ContainerEntity] = set()
        while stack:
            current = stack.pop()
            if current is self:
                return True
            if current in visited:
                continue
            visited.add(current)
            for child in current.children:
                if isinstance(child, ContainerEntity):
                    stack.append(child)
        return False

    def add_child(self, child: Entity) -> None:
        if child is self:
            raise ValueError("Entity cannot be its own child")
        if child in self.children:
            return  # no duplicates

        # no cycle
        if isinstance(child, ContainerEntity) and self._would_form_cycle(child):
            raise ValueError(f"Adding {child.name} would create a cycle")

        self.children.append(child)

    def contains_child(self, child: Entity) -> bool:
        return child in self.children

    def remove_child_if_exists(self, child: Entity) -> None:
        try:
            self.children.remove(child)
        except ValueError:
            pass

    def hasChild(self, entity: Entity) -> bool:
        for child in self.children:
            if child is entity:
                return True
            if isinstance(child, ContainerEntity) and child.hasChild(entity):
                return True
        return False
    
    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:
        info = super().on_perceive(observer, env, depth)

        if(not self.is_any_perceived):
            return info

        count = len(self.children)

        if depth.value <= Depth.MINIMAL.value:
            info["contents_count"] = "unknown"
            return info

        openable = self.get_capability(OpenableCapability)

        if openable and not openable.is_open:
            closed_visibility = max(0.0, openable.visibility_when_closed)
            if closed_visibility > 0:
                reduced_detail = depth.reduced(closed_visibility)
                count_str = Depth.obfuscate_number(reduced_detail, count)
                if count == 0:
                    info["contents_count"] = "empty"
                elif count == 1:
                    info["contents_count"] = f"contains {count_str}"
                else:
                    info["contents_count"] = f"contains {count_str} items"
            else:
                if count == 0:
                    info["contents_count"] = "empty"
                elif count < 3:
                    info["contents_count"] = "something inside"
                elif count < 10:
                    info["contents_count"] = "several things"
                else:
                    info["contents_count"] = "many things"
            return info

        reduced_detail = depth.reduced(self.visibility)
        count_str = Depth.obfuscate_number(reduced_detail, count)

        if count == 0:
            info["contents_count"] = "empty"
        elif count == 1:
            info["contents_count"] = f"contains {count_str}"
        else:
            info["contents_count"] = f"contains {count_str} items"

        if depth.value >= Depth.NORMAL.value and self.children:
            child_detail = reduced_detail.reduced(1)
            info["contents"] = [
                child.on_perceive(observer, env, child_detail)
                for child in self.children
            ]

        return info

class AdvancedContainerEntity(ContainerEntity):
    def __init__(
        self,
        name: str,
        pos: Position | None = None,
        material: str | None = None,
        description: str | None = None,
        uniqueness: float = 0.5,
        prominence: float = 1.0,
        is_collectible: bool = False,
        *,
        is_open: bool = True,
        is_locked: bool = False,
        visibility: float = 1.0,
        visibility_closed: float = 0.0,
    ) -> None:
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible, visibility)

        self._openable = OpenableCapability(
            self,
            initially_open=is_open,
            visibility_when_closed=visibility_closed,
        )
        self.add_capability(self._openable)

        self._lockable = LockableCapability(self, initially_locked=is_locked)
        self.add_capability(self._lockable)

    @property
    def is_open(self) -> bool:
        return self._openable.is_open

    @is_open.setter
    def is_open(self, value: bool) -> None:
        self._openable.is_open = value

    @property
    def is_locked(self) -> bool:
        return self._lockable.is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        self._lockable.is_locked = value

    def add_key(self, entity: Entity) -> None:
        if entity is not None:
            self._lockable.allow_key(entity)

class ConnectorEntity(Entity):
    def __init__(
        self,
        name,
        pos,
        material=None,
        description=None,
        uniqueness=0.5,
        prominence=1,
        is_locked=False,
    ):
        super().__init__(name, pos, material, description, uniqueness, prominence)
        self.otherDoor: ConnectorEntity | None = None
        self.is_locked = is_locked
        self.keys: list[Entity] = []

    def connect(self, entity: "ConnectorEntity") -> None:
        self.otherDoor = entity

    def enter_connect(self, user_entity: Entity) -> None:
        if self.otherDoor is None:
            raise ValueError(f"connectionDoor is None in {self.name}")

        current_room = user_entity.room
        connection_room = self.otherDoor.room
        if current_room is None or connection_room is None:
            raise HardException("room is not registered")
        if current_room == connection_room:
            raise HardException(f"Entity: {user_entity.name} is already in Room: {connection_room.name}")
        user_entity.leave()
        user_entity.pos = self.otherDoor.pos
        user_entity.enter(connection_room)

    def on_interact(self, actor_entity, action: ActionTry) -> str:
        actor_room = actor_entity.room
        if (
            self.room is None
            or actor_room is None
            or not actor_room.contains_entity(self)
        ):
            raise HardException(f"{self.readable_id} is not in your room")

        if action.type in (ActionType.OPEN, ActionType.CLOSE):
            raise SoftException("to go through, use the operator 'USE'")

        if action.type == ActionType.UNLOCK:
            if not self.is_locked:
                return "already unlocked"
            if action.item_1 in self.keys:
                self.is_locked = False
                return "unlocked"
            raise SoftException("wrong key")

        if action.type == ActionType.LOCK:
            if self.is_locked:
                return "already locked"
            if action.item_1 in self.keys:
                self.is_locked = True
                raise SoftException("locked")

        if action.type == ActionType.USE:
            if not self.is_locked:
                actor_entity.use_connector(self)
                return "went through"

        super().on_interact(actor_entity, action)

    def add_key(self, entity: Entity) -> None:
        if (
            isinstance(entity, Entity)
            and entity not in self.keys
        ):
            self.keys.append(entity)


class AgentEntity(Entity):
    def __init__(
        self,
        name: str,
        perception: ObserverPerception,
        pos: Position | None = None,
        material: str | None = None,
        description: str | None = None,
        uniqueness: float = 0.5,
        prominence: float = 1.0,
        is_collectible: bool = False,
    ) -> None:
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible)
        self.perception = perception
        self.inventory: List[Entity] = []

    def take(self, item: Entity) -> None:
        if item is None:
            raise HardException("item does not exist")

        if not item.is_collectible:
            raise SoftException(f"{item.readable_id} can't be collected")

        if self.room is None or self.room is not item.room:
            raise HardException(f"{item.readable_id} is not in your room")

        item.leave()
        item.pos = None
        item.is_collectible = False
        self.inventory.append(item)

    def drop(self, item: Entity) -> None:
        if item not in self.inventory:
            raise HardException(f"{item.readable_id} is not in your inventory")

        room = self.room
        if room is None:
            raise HardException("agent is not in a room")

        item.pos = self.pos
        item.is_collectible = True
        item.enter(room)
        self.inventory.remove(item)

    def take_from(self, item: Entity, container: Entity) -> None:
        if item is None or container is None:
            raise HardException("item or container does not exist")

        if not item.is_collectible:
            raise SoftException(f"{item.readable_id} can't be collected")

        if not isinstance(container, ContainerEntity):
            raise SoftException(f"you can't take something from {container.readable_id}")

        room = self.room
        if room is None or not room.contains_entity(container):
            raise HardException(f"container: {container.readable_id} is not in your room")
        if not container.contains_child(item):
            raise HardException(f"item: {item.readable_id} is not in container: {container.readable_id}")

        item.is_collectible = False
        container.remove_child_if_exists(item)
        self.inventory.append(item)

    def drop_into(self, item: Entity, container: Entity) -> None:
        if item is None or container is None:
            raise HardException("item or container does not exist")

        if not isinstance(container, ContainerEntity):
            raise SoftException(f"you can't drop something into {container.readable_id}")

        if item not in self.inventory:
            raise HardException(f"item: {item.readable_id} is not in inventory")

        item.is_collectible = True
        container.add_child(item)
        self.inventory.remove(item)

    def move_to_object(self, target: Entity) -> None:
        room = self.room

        if room is None:
            raise HardException("agent is not in a room")

        if target.room is room:
            if target.pos is None:
                raise SoftException(f"{target.readable_id} has no accessible position")
            self.move_to_position(target.pos)
            return

        for entity in room.entities:
            if entity.hasChild(target):
                if entity.pos is None:
                    raise SoftException(f"{entity.readable_id} has no accessible position")
                self.move_to_position(entity.pos)
                return

        raise HardException(f"object: {target.readable_id} is not in your room")

    def move_to_position(self, pos) -> None:
        room = self.room
        if room is None:
            raise HardException("agent is not in a room")
        if not room.isPosInRoom(pos):
            raise SoftException(f"You can't go past the wall in room: {room.name}")
        self.pos = pos

    def use_connector(self, connector: Entity) -> None:
        if not isinstance(connector, ConnectorEntity):
            raise SoftException(f"{connector.readable_id} can't be used as a door")

        room = self.room
        if room is None or not room.contains_entity(connector):
            raise HardException(f"connector: {connector.readable_id} is not in your room")

        connector.enter_connect(self)

    def get_inventory(self):
        o = []
        for item in self.inventory:
            data = {}
            data["name"] = item.name
            data["id"] = item.readable_id
            o.append(data)
        return o