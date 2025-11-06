from typing import Set
from uuid import UUID

from enviroment.interaction import Depth, Interaction, PerceptionEnviroment
from enviroment.position import Position
from enviroment.world import World

class Room:
    def __init__(
        self,
        name: str | None,
        extend_x: float = 4.0,
        extend_y: float = 4.0,
        material: str | None = None,
        description: str | None = None,
        uniqueness: float = 0.5,
        light_level: float = 0.5,
        ambient_noise: float = 0.5,
        ambient_smell: float = 0.5
    ) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.extend_x = float(extend_x)
        self.extend_y = float(extend_y)
        self.material = material
        self.description = description
        self.uniqueness = float(uniqueness)
        self.light_level = float(light_level)
        self.ambient_noise = float(ambient_noise)
        self.ambient_smell = float(ambient_smell)
        self.entities: Set[UUID] = set()

        self.uuid: UUID | None = None
        World.add_room(self)

    def perceive(self, observer_uuid, depth: Depth):
        perceptions = []

        for obj in self.entities:
            if obj == observer_uuid:
                continue

            target = World.get_entity(obj)
            observer = World.get_entity(observer_uuid)

            distance: float = observer.pos.distanceTo(target.pos)

            env = PerceptionEnviroment(
                distance_m=distance,
                light_level=self.light_level,
                ambient_noise=self.ambient_noise,
                ambient_smell=self.ambient_smell,
                interactions=[Interaction.INSPECT, Interaction.FEEL]
            )

            perception_data = target.on_perceive(observer.perception, env, depth)

            # ensure it's a dict before appending
            if isinstance(perception_data, dict):
                perceptions.append(perception_data)
            else:
                perceptions.append({"object": str(target), "data": str(perception_data)})

        return perceptions

    def isUuidIRoom(self, uuid) -> bool:
        if uuid in self.entities:
            return True

        for ent in self.entities:
            if World.get_entity(ent).hasChild(uuid):
                return True
            
        return False
    
    def isPosInRoom(self, pos: Position) -> bool:
        if 0 <= pos.x <= self.extend_x and 0 <= pos.y <= self.extend_y:
            return True
        else:
            return False

    # ---------- factory methods ----------
    @classmethod
    def chamber(cls, name: str = "chamber") -> "Room":
        return cls(name, extend_x=1.0, extend_y=3.0)

    @classmethod
    def corridor(cls, length: float, name: str = "corridor") -> "Room":
        return cls(name, extend_x=length, extend_y=2.0)
