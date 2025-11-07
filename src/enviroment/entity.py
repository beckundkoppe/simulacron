
from typing import List
from uuid import UUID

from enviroment.action import ActionTry, ActionType
from enviroment.exception import HardException, SoftException, SuccessException
import config as config
from enviroment.room import Position, Room
from enviroment.world import World
from enviroment.interaction import CompositeDatum, Datum, DatumOperator, Depth, Interaction, PerceptionEnviroment, ObserverPerception, SensoryChannel

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

        self.room: UUID | None = None

        self.uuid: UUID | None = None
        self.readable_id: str | None = None
        World.add_entity(self)

    def enter(self, room: "Entity"):
        if self.room is None:
            newRoom = World.get_room(room.uuid)
            assert newRoom.isPosInRoom(self.pos), f"Position: ({self.pos.x, self.pos.y}) from {self.name} does not fit in Room: {newRoom.name} ({newRoom.extend_x,newRoom.extend_y})"
            
            newRoom.entities.add(self.uuid)
            self.room = room.uuid 
        else:
            print(f"can not enter room {room}: {self.readable_id} is already in a room")

    def leave(self):
        if self.room is not None:
            room = World.get_room(self.room)
            room.entities.remove(self.uuid)
            self.room = None
        else:
            print(f"can not leave room: {self.readable_id} is in no room")

    def hasChild(self, uuid):
        return False

    def on_interact(
        self,
        actor_entity,
        action: ActionTry
    ) -> str:
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

        if self.room is not None:
            self.info["position"] = self.pos.map(self.room).toString()

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
        self.children: List[UUID] = []

    def _would_form_cycle(self, candidate: "ContainerEntity") -> bool:
        stack = [candidate]
        visited: set[UUID] = set()
        while stack:
            current = stack.pop()
            if current.uuid == self.uuid:
                return True
            visited.add(current.uuid)
            for cid in current.children:
                c = World.get_entity(cid) if World else None
                if isinstance(c, ContainerEntity) and c.uuid not in visited:
                    stack.append(c)
        return False

    def add_child(self, child: Entity) -> None:
        if child.uuid == self.uuid:
            raise ValueError("Entity cannot be its own child")
        if child.uuid in self.children:
            return  # no duplicates

        # no cycle
        if isinstance(child, ContainerEntity) and self._would_form_cycle(child):
            raise ValueError(f"Adding {child.name} would create a cycle")

        self.children.append(child.uuid)

    def contains_uuid(self, uuid: UUID) -> None:
        return uuid in self.children

    def remove_child_uuid_if_exists(self, uuid: UUID) -> None:
        try:
            self.children.remove(uuid)
        except ValueError:
            pass

    def hasChild(self, uuid):
        for c in self.children:
            if c == uuid:
                return True
            if World.get_entity(c).hasChild(uuid):
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
                World.get_entity(cid).on_perceive(observer, env, child_detail)
                for cid in self.children
                if World.get_entity(cid)
            ]

        return info

class AdvancedContainerEntity(ContainerEntity):
    def __init__(self, name: str, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0, is_collectible: bool = False,
                 is_open: bool = True, is_locked: bool = False, visibility: float = 1.0, visibility_closed: float = 0.0) -> None:
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible, visibility)

        self.is_open = is_open
        self.is_locked = is_locked
        self.d  = visibility_closed
        self.keys = []

    def on_interact(
        self,
        actor_entity,
        action: ActionTry
    ) -> str:
        print(action)
        if(action.type == ActionType.OPEN):
            if(self.is_locked):
                raise SoftException("cant open, is locked")
            self.is_open = True
            return "opened chest"

        if(action.type == ActionType.CLOSE):
            if(self.is_locked):
                raise SoftException("cant close, is locked")
            self.is_open = False

            return "closed chest"

        if(action.type == ActionType.UNLOCK):
            if(action.item_1 in self.keys):
                self.is_locked = False
                return "unlocked chest"
            
            raise SoftException("wrong key")

        if(action.type == ActionType.LOCK):
            if(action.item_1 in self.keys):
                self.is_locked = True
                raise SoftException("locked chest")
            
            raise SoftException("wrong key")

        super().on_interact(actor_entity, action)
        
    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:
        info = Entity.on_perceive(self, observer, env, depth)

        if(not self.is_any_perceived):
            return info

        # container state
        info["state"] = "open" if self.is_open else "closed"

        count = len(self.children)

        if depth.value <= Depth.MINIMAL.value:
            info["contents_count"] = "unknown"
            return info

        if not self.is_open:
            # --- CLOSED ---
            #reduced_detail = depth.reduced(self.visibility_closed)

            if count == 0:
                info["contents_count"] = "empty"
            elif count < 3:
                # wenige, aber unklar wie viele
                info["contents_count"] = "something inside"
            elif count < 10:
                info["contents_count"] = "several things"
            else:
                info["contents_count"] = "many things"

            # no children perceived
            return info

        # --- OPEN ---
        reduced_detail = depth.reduced(self.visibility)
        count_str = Depth.obfuscate_number(reduced_detail, count)

        if count == 0:
            info["contents_count"] = "empty"
        elif count == 1:
            info["contents_count"] = f"contains {count_str}"
        else:
            info["contents_count"] = f"contains {count_str} items"

        # Bei höheren DetailLevels: Kinder auch auflisten
        if depth.value >= Depth.NORMAL.value and self.children:
            child_detail = reduced_detail.reduced(1)  # Kinder noch eine Stufe unklarer
            info["contents"] = [
                World.get_entity(cid).on_perceive(observer, env, child_detail)
                for cid in self.children
                if World.get_entity(cid)
            ]

        return info
    
    def add_key(self, entity):
        self.keys.append(entity)

class ConnectorEntity(Entity):
    def __init__(self, name, pos, material = None, description = None, uniqueness = 0.5, prominence = 1):
        super().__init__(name, pos, material, description, uniqueness, prominence)
        self.otherDoor : ConnectorEntity = None

    def connect(self, entity: "ConnectorEntity"):
        self.otherDoor = entity

    def enter_connect(self, user_entity: Entity):
        assert self.otherDoor is not None, f"connectionDoor is None in {self.name}"

        currentRoom = World.get_room(user_entity.room)
        connectionRoom = World.get_room(self.otherDoor.room)
        assert currentRoom != connectionRoom, f"Entity: {user_entity.name} is already in Room: {connectionRoom.name}"
        user_entity.leave()
        user_entity.pos = self.otherDoor.pos
        user_entity.enter(connectionRoom)


class AgentEntity(Entity):
    def __init__(self, name: str, perception: ObserverPerception, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0, is_collectible: bool = False,):
        super().__init__(name, pos, material, description, uniqueness, prominence, is_collectible)
        self.perception = perception
        self.inventory: List[UUID] = []

    def take(self, item_uuid: UUID):
        item: Entity = World.get_entity(item_uuid)

        if(not item.is_collectible):
            raise SoftException(f"{item.readable_id} can't be collected")
        
        if(self.room != item.room):
            raise HardException(f"{item.readable_id} is not in your room")

        item.leave()
        item.pos = None
        item.is_collectible = False
        self.inventory.append(item_uuid)

    def drop(self, item_uuid: UUID):
        item: Entity = World.get_entity(item_uuid)
        own_room: Room = World.get_room(self.room)

        if(not(item_uuid in self.inventory)):
            raise HardException(f"{item.readable_id} is not in your inventory")

        item.pos = self.pos
        item.is_collectible = True
        item.enter(own_room)
        self.inventory.remove(item_uuid)

    def take_from(self, item_uuid, container_uuid):
        item = World.get_entity(item_uuid)
        container: ContainerEntity = World.get_entity(container_uuid)

        if(not item.is_collectible):
            raise SoftException(f"{item.readable_id} can't be collected")

        if not isinstance(container, ContainerEntity):
            raise SoftException(f"you can't take something from {container.readable_id}")

        if(not World.get_room(self.room).isUuidIRoom(container_uuid)):
            raise HardException(f"container: {container.readable_id} is not in your room")
        if(not container.contains_uuid(item_uuid)):
            raise HardException(f"item: {item.readable_id} is not in container: {container.readable_id}")

        item.is_collectible = False
        container.remove_child_uuid_if_exists(item_uuid)
        self.inventory.append(item_uuid)

    def drop_into(self, item_uuid, container_uuid):
        item = World.get_entity(item_uuid)
        container: ContainerEntity = World.get_entity(container_uuid)

        if not isinstance(container, ContainerEntity):
            raise SoftException(f"you can't drop something from {container.readable_id}")

        if(not (item_uuid in self.inventory)):
            raise HardException(f"item: {item.readable_id} is not in inventory")

        item.is_collectible = True
        container.add_child(item)
        self.inventory.remove(item_uuid)

    def move_to_object(self, target_uuid: UUID):
        target: Entity = World.get_entity(target_uuid)

        if(self.room != target.room):
            raise HardException(f"object: {target.readable_id} is not in your room")

        self.move_to_position(target.pos)
        
    def move_to_position(self, pos):
        room: Room = World.get_room(self.room)
        if(not room.isPosInRoom(pos)):
            raise SoftException(f"You can't go past the wall in room: {room.name}")
        self.pos = pos
        
    def use_connector(self, connector_uuid: UUID):
        entity = World.get_entity(self.uuid)
        connector = World.get_entity(connector_uuid)

        if(not(isinstance(connector,ConnectorEntity))):
            raise SoftException(f"{connector.readable_id} can't be used as a door")

        if(self.room != connector.room):
            raise HardException(f"connector: {connector.readable_id} is not in your room")

        connector.enter_connect(entity)

    def get_inventory(self):
        o = []
        for item in self.inventory:
            entity = World.get_entity(item)
            data = {}
            data["name"] = entity.name
            data["id"] = entity.readable_id
            o.append(data)
        return o

        

        
