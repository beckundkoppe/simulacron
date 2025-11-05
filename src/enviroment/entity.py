
from typing import List
from uuid import UUID

import config as config
from enviroment.room import Position
from enviroment.world import World
from enviroment.interaction import CompositeDatum, Datum, DatumOperator, Depth, Interaction, PerceptionEnviroment, ObserverPerception, SensoryChannel

class Entity:
    def __init__(self, name: str, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.pos = pos

        self.material = material
        self.description = description
        self.uniqueness = uniqueness
        self.prominence = prominence
        self.room: UUID | None = None

        self.uuid: UUID | None = None
        self.readable_id: str | None = None
        World.add_entity(self)

    def enter(self, room):

        if self.room is None:
            newRoom = World.get_room(room.uuid)
            assert newRoom.isPosInRoom(self.pos), f"Position: ({self.pos.x, self.pos.y}) from {self.name} does not fit in Room: {newRoom.name} ({newRoom.extend_x,newRoom.extend_y})"
                
            newRoom.entities.add(self.uuid)
            self.room = room 
        else:
            print(f"can not enter room {room}: {self.readable_id} is already in a room")

    #def leave(self):
    #    if self.room is not None:
    #        self.room.entities.remove(self.uuid)
    #        self.room = None
    #    else:
    #        print(f"can not leave room: {self.readable_id} is in no room")

    def on_interact(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:
        return self.on_perceive(self, observer, env, depth)

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

        if(not condition()): 
            self.info["id"] = "not uniquely identifiable at a glance"

        if(self.info.get("name" is not None) or self.info.get("object" is not None)):
            self.entity.info["position"] = self.pos.toString()

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
    def __init__(self, name: str, pos: Position | None = None, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0,
                 is_open: bool = True, is_locked: bool = False, visibility_open: float = 1.0, visibility_closed: float = 0.0) -> None:
        super().__init__(name, pos, material, description, uniqueness, prominence)

        self.is_open = is_open
        self.is_locked = is_locked
        self.visibility_open = visibility_open
        self.visibility_closed = visibility_closed
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

    def remove_child_uuid_if_exists(self, uuid: UUID) -> None:
        try:
            self.children.remove(uuid)
        except ValueError:
            pass

    def on_interact(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:

        if(env.interactions == Interaction.OPEN):
            self.is_open = True

        if(env.interactions == Interaction.CLOSE):
            self.is_open = False

        return self.on_perceive(self, observer, env, depth)
        
    def on_perceive(
        self,
        observer: ObserverPerception,
        env: PerceptionEnviroment,
        depth: Depth,
    ) -> dict[str, object]:
        info = super().on_perceive(observer, env, depth)

        if(self.info.get("name" is None)):
            return info

        # container state
        info["is_open"] = "open" if self.is_open else "closed"

        count = len(self.children)

        if depth.value <= Depth.MINIMAL.value:
            info["contents_count"] = "unknown"
            return info

        if not self.is_open:
            # --- CLOSED ---
            reduced_detail = depth.reduced(self.visibility_closed)

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
        reduced_detail = depth.reduced(self.visibility_open)
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


class ConnectorEntity(Entity):
    def __init__(self, name, pos, roomTo, material = None, description = None, uniqueness = 0.5, prominence = 1):
        super().__init__(name, pos, material, description, uniqueness, prominence)
        self.roomTo : UUID = roomTo

    def connect(self, entity: Entity):
        assert entity.room != self.roomTo, "Entity is already in Room: " + self.roomTo.name
        entity.enter(self.roomTo)





        
