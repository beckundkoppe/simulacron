
from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING, Type, TypeVar

from enviroment.action import ActionTry, ActionType
from enviroment.capabilities import LockableCapability, OpenableCapability
from enviroment.exception import HardException, SoftException
import config as config
from enviroment.room import Position, Room
from enviroment.world import World
from enviroment.perception import Datum, DetailLevel, PerceptionEnviroment

if TYPE_CHECKING:
    from enviroment.capabilities import Capability

T = TypeVar("T", bound="Capability")

class Entity:
    def __init__(self, name: str, pos: Position | None = None, description: str | None = None, is_collectible: bool = True) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.pos = pos
        self.description = description
        self.is_collectible = is_collectible

        self.room: Room | None = None

        self.readable_id: str | None = None

        self._capabilities: list[Capability] = []

        World.add_entity(self)


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

    def _distance_to_actor(self, actor_entity: "Entity") -> float | None:
        # Returns Euclidean distance if both positions exist, else None.
        if actor_entity is None:
            return None

        actor_room = actor_entity.room
        root_entity = actor_room.get_root_entity(self) if actor_room else None

        if (
            root_entity is None
            or root_entity.pos is None
            or actor_entity.pos is None
        ):
            return None

        return root_entity.pos.distanceTo(actor_entity.pos)

    @staticmethod
    def _format_position_for_agent(pos: Position, room: Room | None) -> str:
        active_config = getattr(config, "CONFIG", None)

        if room is None or active_config is None:
            return f"({pos.x:.1f}, {pos.y:.1f})"

        mapped = pos.map(room)
        if mapped.type == config.PositionType.CHESSBOARD:
            return mapped.toString()
        if mapped.type == config.PositionType.ROOMLESS:
            return mapped.toString()
        return f"({mapped.x:.1f}, {mapped.y:.1f})"

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
        if dist > config.INTERACTION_DISTANCE:
            raise SoftException(
                f"{self.readable_id} is too far away to interact.",
                console_message=(
                    f"Range check failed: distance={dist:.3f}, "
                    f"limit={config.INTERACTION_DISTANCE:.3f}."
                ),
                hint="Move closer to the object before interacting.",
                context={
                    "actor": getattr(actor_entity, "readable_id", None),
                    "target": self.readable_id,
                    "distance": dist,
                    "limit": config.INTERACTION_DISTANCE,
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
                f"Position: {self._format_position_for_agent(self.pos, room)} from {self.name} "
                f"does not fit in Room:{room.name} ({room.extend_x,room.extend_y})"
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

    def on_interact(self, actor_entity, action: ActionTry) -> str:
        #if self not in actor_entity.get_inventory():
        #    raise SoftException(
        #        "You can only drop items that are in your inventory.",
        #        console_message=(
        #            f"The object '{self.readable_id}' is not in your inventory."
        #        ),
        #        hint="Select an item from your inventory if you want to drop it.",
        #        context={
        #            "entity": self.readable_id,
        #            "inventory": [e.readable_id for e in actor_entity.get_inventory()],
        #        },
        #    )


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

    def on_perceive(self, observer:"Entity", env: PerceptionEnviroment, depth: DetailLevel) -> dict[str, object]:
    
        self.info: dict[str, object] = {}

        name = Datum(key="name", value=self.name, info=self.info)
        description = Datum(key="description", value=self.description, info=self.info)
        id = Datum(key="id", value=self.readable_id, info=self.info)

        all_datums: list[Datum] = [name]
        if (self.description): all_datums.append(description)
        all_datums.append(id)

        for dat in all_datums:
            dat.perceive(observer, env, depth)

        self.is_any_perceived = self.info.get("name") is not None

        if(self.info.get("id") is None): 
            self.info["id"] = "not uniquely identifiable at a glance"

        if self.room is not None:
            self.info["position"] = self.pos.map(self.room).toString()

        for capability in self._capabilities:
            capability.on_perceive(observer, env, depth, self.info)

        return self.info
    

class ContainerEntity(Entity):
    def __init__(self, name: str, pos: Position | None = None, description: str | None = None, is_collectible: bool = False) -> None:
        super().__init__(name, pos, description, is_collectible)

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
    
    def on_perceive(self, observer:Entity, env: PerceptionEnviroment, detail: DetailLevel) -> dict[str, object]:
        info = super().on_perceive(observer, env, detail)

        if(not self.is_any_perceived):
            return info

        count = len(self.children)

        if detail <= DetailLevel.BAD:
            info["contents_count"] = "unknown"
            return info

        openable = self.get_capability(OpenableCapability)
        
        count_str = DetailLevel.obfuscate_number(detail, count)
        if count == 0:
            info["contents_count"] = "empty"
        elif count == 1:
            info["contents_count"] = f"contains {count_str}"
        else:
            info["contents_count"] = f"contains {count_str} items"

        if (openable and openable.is_open) or not openable:
            if detail >= DetailLevel.NORMAL and self.children:
                child_detail = DetailLevel.reduce(detail)
                info["contents"] = [
                    child.on_perceive(observer, env, child_detail)
                    for child in self.children
                ]
        else:
            info["contents"] = ["unknown"]

        return info


class AdvancedContainerEntity(ContainerEntity):
    def __init__(self, name: str, pos: Position | None = None, description: str | None = None, is_collectible: bool = False, is_open: bool = True, is_locked: bool = False, visibility_closed=0.0) -> None:
        super().__init__(name, pos, description, is_collectible)

        self._openable = OpenableCapability(
            self,
            initially_open=is_open,
            visibility_closed=visibility_closed,
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

    def register_key(self, entity: Entity) -> None:
        if entity is not None:
            self._lockable.allow_key(entity)


class ConnectorEntity(Entity):
    def __init__(self, name, pos, description = None, is_locked = False):
        super().__init__(name, pos, description, is_collectible=False)
        self.rooms: dict[Room, Position] = {}
        self.keys: set[Entity] = set()

        self._openable = OpenableCapability(self, initially_open=True)
        self.add_capability(self._openable)

        self._lockable = LockableCapability(self, initially_locked=is_locked)
        self.add_capability(self._lockable)

    @property
    def is_locked(self) -> bool:
        return self._lockable.is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        self._lockable.is_locked = value

    def add_key(self, entity: Entity):
        self.keys.add(entity)
        self._lockable.allow_key(entity)

    def connect(self, entity: "ConnectorEntity"):
        self.otherDoor = entity

    def add_room(self, room: Room, pos: Position) -> None:
        assert room.isPosInRoom(pos), (
            f"Position: {self._format_position_for_agent(pos, room)} from {self.name} "
            f"does not fit in Room:{room.name} ({room.extend_x,room.extend_y})"
        )
        self.rooms[room] = pos
        room.entities.add(self)
        if self.room is None:
            self.room = room
        self.pos = pos

    def enter(self, room: Room, pos: Position | None = None):
        position = pos if pos is not None else self.pos
        if position is None:
            raise ValueError("Connector needs a position per room")
        self.add_room(room, position)

    def prepare_for_perception(self, room: Room) -> None:
        if room in self.rooms:
            self.pos = self.rooms[room]
            self.room = room

    def get_position_for_room(self, room: Room) -> Position | None:
        return self.rooms.get(room)

    def get_other_room(self, room: Room | None) -> Room | None:
        if room is None:
            return None
        for candidate in self.rooms:
            if candidate is not room:
                return candidate
        return None

    def _distance_to_actor(self, actor_entity: "Entity") -> float | None:
        actor_room = actor_entity.room if actor_entity else None
        if actor_room is None:
            return None
        connector_pos = self.get_position_for_room(actor_room)
        if connector_pos is None or actor_entity.pos is None:
            return None
        return connector_pos.distanceTo(actor_entity.pos)

    def _ensure_same_room(self, actor_entity: "Entity") -> None:
        actor_room = actor_entity.room if actor_entity else None
        if actor_room is None or actor_room not in self.rooms:
            raise HardException(
                f"{self.readable_id} is not in your current room.",
                console_message=(
                    f"{getattr(actor_entity, 'readable_id', 'unknown')} attempted to interact with "
                    f"'{self.readable_id}', but it is only reachable from "
                    f"'{', '.join(room.readable_id for room in self.rooms)}'."
                ),
                hint="Move to a room connected to the door before interacting.",
                context={
                    "actor": getattr(actor_entity, "readable_id", None),
                    "actor_room": actor_room.readable_id if actor_room else None,
                    "target": self.readable_id,
                },
            )

    def enter_connect(self, user_entity: Entity):
        destination_room = self.get_other_room(user_entity.room)
        if destination_room is None:
            raise HardException(
                "This connector is not linked to another room.",
                console_message=(
                    f"Connector '{self.readable_id}' is missing a destination from "
                    f"'{user_entity.room.readable_id if user_entity.room else 'unknown'}'."
                ),
                hint="Ensure doors are connected to two rooms before traversal.",
            )

        target_position = self.get_position_for_room(destination_room)
        assert target_position is not None, "destination position missing"

        user_entity.leave()
        user_entity.pos = target_position
        user_entity.enter(destination_room)

    def on_interact(self, actor_entity, action: ActionTry) -> str:
        self.prepare_for_perception(actor_entity.room)
        self._ensure_same_room(actor_entity)
        self._ensure_in_range(actor_entity)

        if action.type == ActionType.LOOK_THROUGH:
            if not self._openable.is_open:
                raise SoftException(
                    "You need to open the door before looking through it.",
                    console_message=(
                        f"Attempted LOOK_THROUGH on closed connector '{self.readable_id}'."
                    ),
                    hint="Use the OPEN operator first.",
                )

            destination_room = self.get_other_room(actor_entity.room)
            if destination_room is None:
                raise HardException(
                    "This door does not lead anywhere.",
                    console_message=(
                        f"Connector '{self.readable_id}' has no destination room registered."
                    ),
                    hint="Connect the door to a neighbouring room before looking through.",
                )

            observations = []
            origin_position = self.get_position_for_room(actor_entity.room)
            for obj in destination_room.entities:
                if obj is actor_entity:
                    continue
                obj.prepare_for_perception(destination_room) if hasattr(obj, "prepare_for_perception") else None
                target_pos = obj.pos
                if origin_position is None or target_pos is None:
                    continue
                env = PerceptionEnviroment(
                    distance_m=origin_position.distanceTo(target_pos),
                    light_level=destination_room.light_level,
                )
                observations.append(
                    obj.on_perceive(actor_entity, env, DetailLevel.NORMAL)
                )

            return (
                f"Through {self.readable_id} you see into {destination_room.readable_id}: "
                f"{observations or 'nothing of interest'}"
            )

        if action.type in (ActionType.OPEN, ActionType.CLOSE):
            result = self._openable.on_interact(actor_entity, action)
            if result is not None:
                return result

        if action.type in (ActionType.UNLOCK, ActionType.LOCK):
            if action.item_1 is not None:
                self._lockable.allow_key(action.item_1)
            result = self._lockable.on_interact(actor_entity, action)
            if result is not None:
                return result

        if action.type == ActionType.USE:
            if self._lockable.is_locked:
                raise SoftException(
                    "The door is locked.",
                    console_message=(
                        f"Use prevented because '{self.readable_id}' is locked."
                    ),
                    hint="Unlock it before trying to go through.",
                )
            if not self._openable.is_open:
                raise SoftException(
                    "Open the door before going through.",
                    console_message=(
                        f"Use prevented because '{self.readable_id}' is closed."
                    ),
                    hint="Use the OPEN operator first.",
                )
            destination_room = self.get_other_room(actor_entity.room)
            actor_entity.use_connector(self)
            destination_name = destination_room.name if destination_room else "the next room"
            return f"Went through {self.readable_id} into {destination_name}"

        return super().on_interact(actor_entity, action)

class AgentEntity(Entity):
    def __init__(self, name: str, pos: Position | None = None, description: str | None = None, is_collectible: bool = False,):
        super().__init__(name, pos, description, is_collectible)
        self.inventory: List[Entity] = []

    def take(self, item: Entity):
        item._ensure_in_range(self)

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
        item._ensure_same_room(self)
        item._ensure_in_range(self)

        if not item.is_collectible:
            raise SoftException(
                f"{item.readable_id} cannot be collected.",
                console_message=(
                    f"Attempted to take non-collectible '{item.readable_id}' from container "
                    f"'{container.readable_id}'."
                ),
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
            )
        if not container.contains_entity(item):
            raise HardException(
                f"{item.readable_id} is not inside {container.readable_id}.",
                console_message=(
                    f"Removal failed; container '{container.readable_id}' does not hold '{item.readable_id}'."
                ),
            )

        item.is_collectible = False
        container.remove_child_if_exists(item)
        self.inventory.append(item)

    def drop_into(self, item: Entity, container: ContainerEntity):
        container._ensure_same_room(self)
        container._ensure_in_range(self)

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

        if target in self.inventory:
            raise SoftException(
                "You cannot move to an item that is inside your inventory.",
                console_message=(
                    f"The target '{target.readable_id}' is already in your inventory."
                ),
                hint="Choose an object in the room if you want to move toward it.",
                context={"target": target.readable_id},
            )

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
                    f"Agent '{self.readable_id}' attempted to move to position "
                    f"{self._format_position_for_agent(pos, room)} without being in a room."
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
                    f"Requested position {self._format_position_for_agent(pos, room)} lies outside room "
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
        if self.room not in connector.rooms:
            agent_room = self.room
            raise HardException(
                f"Connector {connector.readable_id} is not in your current room.",
                console_message=(
                    f"Agent '{self.readable_id}' attempted to use connector '{connector.readable_id}' while staying in "
                    f"'{agent_room.name if agent_room else 'unknown'}'."
                ),
                hint="Move to a room that contains the door before using it.",
                context={
                    "agent": self.readable_id,
                    "agent_room": agent_room.readable_id if agent_room else None,
                    "connector": connector.readable_id,
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

def connect_rooms_with_door(room_a, pos_a, room_b, pos_b, description = None):
    # create shared door
    door = ConnectorEntity("door", pos_a, description=description)

    # assign to rooms
    door.enter(room_a, pos_a)
    door.enter(room_b, pos_b)

    return door