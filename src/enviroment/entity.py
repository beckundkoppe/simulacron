
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

    def _interaction_max_distance(self) -> float:
        # Prefer a per-object override; fall back to global config or 1.5.
        return getattr(self, "interaction_distance",
                       getattr(config, "MAX_INTERACTION_DISTANCE", 1.5))

    def _distance_to_actor(self, actor_entity: "Entity") -> float | None:
        # Returns Euclidean distance if both positions exist, else None.
        ownPos = actor_entity.room.get_entity(self).pos

        if actor_entity is None or ownPos is None or actor_entity.pos is None:
            return None
        dx = (ownPos.x - actor_entity.pos.x)
        dy = (ownPos.y - actor_entity.pos.y)
        return (dx * dx + dy * dy) ** 0.5

    def _ensure_in_range(self, actor_entity: "Entity") -> None:
        # Enforce max interaction radius.
        dist = self._distance_to_actor(actor_entity)
        if dist is None:
            raise SoftException(
                "Interaction distance cannot be determined.",
                console_message=(
                    f"Either '{getattr(actor_entity, 'readable_id', 'unknown')}' or "
                    f"'{self.readable_id}' has no position set."
                ),
                hint="Ensure both entities have valid positions before interacting.",
                context={
                    "actor": getattr(actor_entity, "readable_id", None),
                    "target": self.readable_id,
                },
            )
        if dist > self._interaction_max_distance():
            raise SoftException(
                f"{self.readable_id} is too far away to interact.",
                console_message=(
                    f"Range check failed: distance={dist:.3f}, "
                    f"limit={self._interaction_max_distance():.3f}."
                ),
                hint="Move closer to the object before interacting.",
                context={
                    "actor": getattr(actor_entity, "readable_id", None),
                    "target": self.readable_id,
                    "distance": dist,
                    "limit": self._interaction_max_distance(),
                },
            )

    def _ensure_same_room(self, actor_entity: "Entity") -> None:
        actor_room = actor_entity.room if actor_entity else None
        target_room = self.room
        if not actor_room or not actor_room.contains_entity(self):
            raise HardException(
                f"{self.readable_id} is not in your current room.",
                console_message=(
                    f"{getattr(actor_entity, 'readable_id', 'unknown')} attempted to interact with "
                    f"'{self.readable_id}', but it resides in "
                    f"'{target_room.name if target_room else 'unknown'}' while the actor is in "
                    f"'{actor_room.name if actor_room else 'unknown'}'."
                ),
                hint="Move to the object's room before interacting with it.",
                context={
                    "actor": getattr(actor_entity, "readable_id", None),
                    "actor_room": actor_room.readable_id if actor_room else None,
                    "target": self.readable_id,
                    "target_room": target_room.readable_id if target_room else None,
                },
            )

    def enter(self, room: Room):
        if self.room is None:
            assert room.isPosInRoom(self.pos), (
                f"Position: ({self.pos.x, self.pos.y}) from {self.name} does not fit in Room:{room.name}"
                f" ({room.extend_x,room.extend_y})"
            )

            room.entities.add(self)
            self.room = room
        else:
            print(f"can not enter room {room}: {self.readable_id} is already in a room")

    def leave(self):
        if self.room is not None:
            room = self.room
            room.entities.remove(self)
            self.room = None
        else:
            print(f"can not leave room: {self.readable_id} is in no room")

    def hasChild(self, entity):
        return False

    def on_interact(
        self,
        actor_entity,
        action: ActionTry
    ) -> str:
        self._ensure_same_room(actor_entity)
        self._ensure_in_range(actor_entity)

        for capability in self._capabilities:
            if capability.supports(action.type):
                result = capability.on_interact(actor_entity, action)
                if result is not None:
                    return result

        raise SoftException(
            "That interaction had no effect on this object.",
            console_message=(
                f"No capability on '{self.readable_id}' handled action "
                f"'{getattr(action.type, 'value', str(action.type))}'."
            ),
            hint="Try a different operator or interact with another object.",
            context={
                "entity": self.readable_id,
                "action": getattr(action.type, "value", str(action.type)),
            },
        )

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

        if self.room is not None:
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
        self.children: List["Entity"] = []

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

    def contains_entity(self, entity: Entity) -> bool:
        return entity in self.children

    def remove_child_if_exists(self, entity: Entity) -> None:
        try:
            self.children.remove(entity)
        except ValueError:
            pass

    def hasChild(self, entity):
        for child in self.children:
            if child is entity:
                return True
            if child.hasChild(entity):
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
    def __init__(self, name, pos, material = None, description = None, uniqueness = 0.5, prominence = 1, is_locked = False):
        super().__init__(name, pos, material, description, uniqueness, prominence)
        self.otherDoor : ConnectorEntity = None
        self.is_locked = is_locked
        self.keys: list[Entity] = []

    def connect(self, entity: "ConnectorEntity"):
        self.otherDoor = entity

    def enter_connect(self, user_entity: Entity):
        assert self.otherDoor is not None, f"connectionDoor is None in {self.name}"

        current_room = user_entity.room
        connection_room = self.otherDoor.room
        assert current_room != connection_room, f"Entity: {user_entity.name} is already in Room: {connection_room.name}"
        user_entity.leave()
        user_entity.pos = self.otherDoor.pos
        user_entity.enter(connection_room)

    def on_interact(
        self,
        actor_entity,
        action: ActionTry
    ) -> str:

        self._ensure_same_room(actor_entity)
        self._ensure_in_range(actor_entity)

        if(action.type == ActionType.OPEN or action.type == ActionType.CLOSE):
            raise SoftException(
                "Use the USE operator to go through this connector.",
                console_message=(
                    f"Connector '{self.readable_id}' received '{action.type.value}' from "
                    f"'{getattr(actor_entity, 'readable_id', 'unknown')}'."
                ),
                hint="Invoke interact_with_object(..., operator=\"GO_THROUGH\") to traverse connectors.",
                context={
                    "connector": self.readable_id,
                    "operator": action.type.value,
                },
            )

        if(action.type == ActionType.UNLOCK):
            if(not self.is_locked):
                return "already unlocked" #war hier <----
            if(action.item_1 in self.keys):
                self.is_locked = False
                return "unlocked"

            key_id = getattr(action.item_1, "readable_id", None)

            raise SoftException(
                "That key does not fit this lock.",
                console_message=(
                    f"Key '{key_id or 'unknown'}' failed to unlock connector '{self.readable_id}'."
                ),
                hint="Use a matching key before unlocking.",
                context={
                    "connector": self.readable_id,
                    "key": key_id,
                },
            )

        if(action.type == ActionType.LOCK):
            if(self.is_locked):
                return "already locked"
            if(action.item_1 in self.keys):
                self.is_locked = True
                key_id = getattr(action.item_1, "readable_id", None)
                raise SoftException(
                    "The connector is now locked.",
                    console_message=(
                        f"Connector '{self.readable_id}' locked using key '{key_id or 'unknown'}'."
                    ),
                    hint="Unlock it again before attempting to use it.",
                    context={
                        "connector": self.readable_id,
                        "key": key_id,
                    },
                )

        if(action.type == ActionType.USE):
            if not self.is_locked:
                actor_entity.use_connector(self)
                return "went through"
            
        super().on_interact(actor_entity, action)

    def add_key(self, entity):
        self.keys.append(entity)

class AgentEntity(Entity):
    def __init__(self, name: str, perception: ObserverPerception, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0, is_collectible: bool = False,):
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible)
        self.perception = perception
        self.inventory: List[Entity] = []

    def take(self, item: Entity):
        if not item.is_collectible:
            raise SoftException(
                f"{item.readable_id} cannot be collected.",
                console_message=(
                    f"Attempted to collect non-collectible entity '{item.readable_id}'."
                ),
                hint="Target objects marked as collectible in your observation.",
                context={
                    "item": item.readable_id,
                    "action": "take",
                },
            )

        if self.room != item.room:
            agent_room = self.room
            item_room = item.room
            raise HardException(
                f"{item.readable_id} is not in your current room.",
                console_message=(
                    f"Agent '{self.readable_id}' tried to take '{item.readable_id}' from "
                    f"'{item_room.name if item_room else 'unknown'}' while located in "
                    f"'{agent_room.name if agent_room else 'unknown'}'."
                ),
                hint="Move to the room containing the item before taking it.",
                context={
                    "agent": self.readable_id,
                    "agent_room": agent_room.readable_id if agent_room else None,
                    "item": item.readable_id,
                    "item_room": item_room.readable_id if item_room else None,
                },
            )

        item.leave()
        item.pos = None
        item.is_collectible = False
        self.inventory.append(item)

    def drop(self, item: Entity):
        own_room: Room = self.room

        if item not in self.inventory:
            inventory_ids = [
                ent.readable_id
                for ent in self.inventory
                if ent and ent.readable_id
            ]
            raise HardException(
                f"{item.readable_id} is not in your inventory.",
                console_message=(
                    f"Drop request for '{item.readable_id}' failed; inventory contains: "
                    f"{', '.join(inventory_ids) or 'nothing'}."
                ),
                hint="Check your inventory in the observation before dropping items.",
                context={
                    "item": item.readable_id,
                    "inventory": inventory_ids,
                },
            )

        item.pos = self.pos
        item.is_collectible = True
        if own_room is not None:
            item.enter(own_room)
        self.inventory.remove(item)

    def take_from(self, item: Entity, container: ContainerEntity):
        if not item.is_collectible:
            raise SoftException(
                f"{item.readable_id} cannot be collected.",
                console_message=(
                    f"Attempted to take non-collectible '{item.readable_id}' from container "
                    f"'{container.readable_id}'."
                ),
                hint="Only collectible items can be removed from containers.",
                context={
                    "item": item.readable_id,
                    "container": container.readable_id,
                    "action": "take_from",
                },
            )

        if not isinstance(container, ContainerEntity):
            raise SoftException(
                f"You can't take items from {container.readable_id}.",
                console_message=(
                    f"Take-from requested on non-container '{container.readable_id}'."
                ),
                hint="Select a valid container or use TAKE for items on the floor.",
                context={
                    "item": item.readable_id,
                    "container": container.readable_id,
                },
            )

        agent_room = self.room
        container_room = container.room
        if not agent_room or not agent_room.contains_entity(container):
            raise HardException(
                f"Container {container.readable_id} is not in your current room.",
                console_message=(
                    f"Agent '{self.readable_id}' attempted to take '{item.readable_id}' from "
                    f"'{container.readable_id}' located in "
                    f"'{container_room.name if container_room else 'unknown'}'."
                ),
                hint="Move to the room that holds the container before interacting with it.",
                context={
                    "agent": self.readable_id,
                    "agent_room": agent_room.readable_id if agent_room else None,
                    "container": container.readable_id,
                    "container_room": container_room.readable_id if container_room else None,
                },
            )
        if not container.contains_entity(item):
            raise HardException(
                f"{item.readable_id} is not inside {container.readable_id}.",
                console_message=(
                    f"Removal failed; container '{container.readable_id}' does not hold '{item.readable_id}'."
                ),
                hint="Check the container contents from your observation before taking an item.",
                context={
                    "container": container.readable_id,
                    "requested_item": item.readable_id,
                    "contents": [
                        child.readable_id
                        for child in getattr(container, "children", [])
                        if getattr(child, "readable_id", None)
                    ],
                },
            )

        item.is_collectible = False
        container.remove_child_if_exists(item)
        self.inventory.append(item)

    def drop_into(self, item: Entity, container: ContainerEntity):
        if not isinstance(container, ContainerEntity):
            raise SoftException(
                f"You can't place items into {container.readable_id}.",
                console_message=(
                    f"Drop-into attempted on non-container '{container.readable_id}'."
                ),
                hint="Choose a valid container or drop the item on the floor.",
                context={
                    "item": item.readable_id,
                    "target": container.readable_id,
                },
            )

        if item not in self.inventory:
            inventory_ids = [
                ent.readable_id
                for ent in self.inventory
                if ent and ent.readable_id
            ]
            raise HardException(
                f"{item.readable_id} is not in your inventory.",
                console_message=(
                    f"Drop-into failed; inventory currently contains: "
                    f"{', '.join(inventory_ids) or 'nothing'}."
                ),
                hint="Take the item first before trying to place it elsewhere.",
                context={
                    "item": item.readable_id,
                    "inventory": inventory_ids,
                },
            )

        item.is_collectible = True
        container.add_child(item)
        self.inventory.remove(item)

    def move_to_object(self, target: Entity):
        room: Room | None = self.room

        if room is None:
            raise HardException(
                "You are not currently inside a room.",
                console_message=(
                    f"Agent '{self.readable_id}' attempted to move towards '{target.readable_id}' without an assigned room."
                ),
                hint="Ensure the agent has entered a room before issuing movement commands.",
                context={
                    "agent": self.readable_id,
                    "target": target.readable_id,
                },
            )

        if room == target.room:
            self.move_to_position(target.pos)
            return
        else:
            for ent in room.entities:
                if ent.hasChild(target):
                     self.move_to_position(ent.pos)
                     return

        agent_room = self.room
        target_room = target.room
        raise HardException(
            f"{target.readable_id} is not reachable from your current room.",
            console_message=(
                f"Agent '{self.readable_id}' attempted to move to '{target.readable_id}' located in "
                f"'{target_room.name if target_room else 'unknown'}' while staying in "
                f"'{agent_room.name if agent_room else 'unknown'}'."
            ),
            hint="Travel to the room containing the target before moving directly to it.",
            context={
                "agent": self.readable_id,
                "agent_room": agent_room.readable_id if agent_room else None,
                "target": target.readable_id,
                "target_room": target_room.readable_id if target_room else None,
            },
        )

    def move_to_position(self, pos):
        room: Room | None = self.room
        if room is None:
            raise HardException(
                "You are not currently inside a room.",
                console_message=(
                    f"Agent '{self.readable_id}' attempted to move to position ({pos.x:.1f}, {pos.y:.1f}) without being in a room."
                ),
                hint="Ensure the agent has entered a room before moving.",
                context={
                    "agent": self.readable_id,
                    "requested_position": {"x": pos.x, "y": pos.y},
                },
            )
        if not room.isPosInRoom(pos):
            raise SoftException(
                f"You can't move past the walls of {room.name}.",
                console_message=(
                    f"Requested position ({pos.x:.1f}, {pos.y:.1f}) lies outside room "
                    f"'{room.name}' with bounds x<= {room.extend_x}, y<= {room.extend_y}."
                ),
                hint="Stay within the room dimensions reported in the observation.",
                context={
                    "room": room.readable_id,
                    "requested_position": {"x": pos.x, "y": pos.y},
                    "bounds": {"extend_x": room.extend_x, "extend_y": room.extend_y},
                },
            )
        self.pos = pos

    def use_connector(self, connector: ConnectorEntity):
        if not isinstance(connector, ConnectorEntity):
            raise SoftException(
                f"{getattr(connector, 'readable_id', 'This object')} cannot be used as a door.",
                console_message=(
                    f"use_connector expected a ConnectorEntity but received {type(connector).__name__}."
                ),
                hint="Target an actual connector (doorway) before calling USE.",
                context={
                    "requested": getattr(connector, "readable_id", None),
                    "expected_type": "ConnectorEntity",
                },
            )

        if self.room != connector.room:
            agent_room = self.room
            connector_room = connector.room
            raise HardException(
                f"Connector {connector.readable_id} is not in your current room.",
                console_message=(
                    f"Agent '{self.readable_id}' attempted to use connector '{connector.readable_id}' located in "
                    f"'{connector_room.name if connector_room else 'unknown'}' while staying in "
                    f"'{agent_room.name if agent_room else 'unknown'}'."
                ),
                hint="Move to the room where the connector resides before using it.",
                context={
                    "agent": self.readable_id,
                    "agent_room": agent_room.readable_id if agent_room else None,
                    "connector": connector.readable_id,
                    "connector_room": connector_room.readable_id if connector_room else None,
                },
            )

        connector.enter_connect(self)

    def get_inventory(self):
        o = []
        for item in self.inventory:
            data = {}
            data["name"] = item.name
            data["id"] = item.readable_id
            o.append(data)
        return o
