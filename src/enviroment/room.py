from typing import Set, TYPE_CHECKING

from enviroment.interaction import Depth, Interaction, PerceptionEnviroment
from enviroment.position import Position
from enviroment.world import World

if TYPE_CHECKING:
    from enviroment.entity import Entity

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
        self.entities: Set["Entity"] = set()

        World.add_room(self)

    def perceive(self, observer_entity: "Entity", depth: Depth):
        perceptions = []

        for obj in self.entities:
            if obj is observer_entity:
                continue

            target = obj
            observer = observer_entity

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

    def contains_entity(self, entity: "Entity") -> bool:
        return self.get_root_entity(entity) is not None
    
    def isPosInRoom(self, pos: Position) -> bool:
        if 0 <= pos.x <= self.extend_x and 0 <= pos.y <= self.extend_y:
            return True
        else:
            return False
        
    def get_root_entity(self, entity: "Entity") -> "Entity":
        if entity in self.entities:
            return entity

        for ent in self.entities:
            if ent.hasChild(entity):
                return ent
            
        return None


    # ---------- factory methods ----------
    @classmethod
    def chamber(cls, name: str = "chamber") -> "Room":
        return cls(name, extend_x=1.0, extend_y=3.0)

    @classmethod
    def corridor(cls, length: float, name: str = "corridor") -> "Room":
        return cls(name, extend_x=length, extend_y=2.0)
