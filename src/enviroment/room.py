from typing import Set, TYPE_CHECKING

from enviroment.perception import DetailLevel, PerceptionEnviroment
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
        description: str | None = None,
        light_level: bool = True
    ) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.extend_x = float(extend_x)
        self.extend_y = float(extend_y)
        self.description = description
        self.light_level: bool = light_level
        self.entities: Set["Entity"] = set()

        World.add_room(self)

    def perceive(self, observer: "Entity", depth: DetailLevel):
        perceptions = []

        for obj in self.entities:
            if obj is observer:
                continue

            target = obj
            observer = observer

            distance: float = observer.pos.distanceTo(target.pos)

            env = PerceptionEnviroment(
                distance_m=distance,
                light_level=self.light_level,
            )

            perception_data = target.on_perceive(observer, env, depth)

            if isinstance(perception_data, dict):
                perceptions.append(perception_data)
            else:
                raise Exception() #TODO
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
    def default(cls, factor: float = 1.0, name: str = "corridor", description: str | None = None, light_level: bool = True) -> "Room":
        return cls(name, extend_x=4*factor, extend_y=4*factor, description=description, light_level=light_level)

    @classmethod
    def chamber(cls, name: str = "chamber", description: str | None = None, light_level: bool = True) -> "Room":
        return cls(name, extend_x=1.0, extend_y=3.0, description=description, light_level=light_level)

    @classmethod
    def corridor(cls, length: float, name: str = "corridor", description: str | None = None, light_level: bool = True) -> "Room":
        return cls(name, extend_x=length, extend_y=2.0, description=description, light_level=light_level)
