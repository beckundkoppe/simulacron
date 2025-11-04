
from uuid import UUID

import config as config
from world import World
from world.position import Position
from world.room import Room
from world.perception import CompositeDatum, Datum, DatumOperator, Depth, PerceptionEnviroment, ObserverPerception, SensoryChannel

class Entity:
    world: World = None

    def __init__(self, name: str, pos: Position, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0) -> None:
        assert isinstance(name, str) and name.strip(), "name must be a non-empty string"
        self.name = name
        self.pos = pos

        self.material = material
        self.description = description

        self.uniqueness = uniqueness
        self.room: Room | None = None,

        self.uuid: UUID | None = None
        self.readable_id: str | None = None
        assert Entity.world is not None, "set Entity.world before creation of entities"
        Entity.world.add(self)

    def enter(self, room: Room):
        if self.room is None:
            room.entities.add(self.uuid)
            self.room = room
        else:
            print(f"can not enter room: {self.readable_id} is already in a room")

    def leave(self):
        if self.room is not None:
            self.room.entities.remove(self.uuid)
            self.room = None
        else:
            print(f"can not leave room: {self.readable_id} is in no room")

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
            d.perceive(config.PARTIAL_DISCOUVERY, observer, env, depth)

        if(not condition()): 
            self.info["id"] = "not uniquely identifiable at a glance"

        if(self.info.get("name" is not None)):
            self.entity.info["position"] = self.pos.toString()

        return self.info

class ContainerEntity(Entity):
    def __init__(self, name: str, material: str | None = None, description: str | None = None, uniqueness: float = 0.5, prominence: float = 1.0) -> None:
        super.__init__(name, material, description, uniqueness, prominence)
