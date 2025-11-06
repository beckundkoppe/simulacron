from dataclasses import dataclass
from enum import Enum
from typing import Callable

from enviroment.entity import AgentEntity, ContainerEntity, ConnectorEntity, Entity
from enviroment.interaction import ObserverPerception
from enviroment.room import Position, Room
from enviroment.world import World


class LevelType(Enum):
    SHORT           = "short horizon"
    LONG            = "long horizon"

@dataclass()
class LevelSpec:
    agent_entities: list[tuple[Entity, str]]
    is_success: Callable[[], bool]

@dataclass()
class Level:
    name: str
    type: LevelType
    optimal_steps: int
    build: Callable[[], LevelSpec]

def buildLevel_Potato() -> LevelSpec:
    main = Room("main", 4, 4)
    chamber = Room.chamber()

    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    tron.enter(main)

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    key.enter(main)
    
    table = ContainerEntity("table", Position(0.0, 3.0))
    table.enter(main)
    
    doorMain = ConnectorEntity("door", Position(4.0,4.0))
    doorMain.enter(main)

    doorChamber = ConnectorEntity("door", Position(0.0,1.0))
    doorChamber.enter(chamber)

    doorMain.connect(doorChamber)
    doorChamber.connect(doorMain)

    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    chest = ContainerEntity("chest")
    chest.add_child(potato)

    box = ContainerEntity("box")
    box.add_child(diamond)

    shelf1 = ContainerEntity("shelf1", Position(0.0,0.0))
    shelf1.enter(chamber)
    shelf1.add_child(chest)

    shelf2 = ContainerEntity("shelf2", Position(0.0,2.0))
    shelf2.enter(chamber)
    shelf2.add_child(box)

    return LevelSpec(
        agent_entities=[
            (tron,"Goal: Find a potato, take it and place it on the table. GIVE the next toolcall - nothing more"),
        ],
        is_success=potato_check_win,
    )

def potato_check_win() -> bool:
    potato = None
    table: ContainerEntity = None

    for uuid in World.entities:
        ent = World.get_entity(uuid)
        if(ent.name == "potato"):
            potato = ent
        if(ent.name == "table"):
            table = ent
    
    assert potato != None, "No potato"
    assert table != None, "No table"
    
    return potato.uuid in table.children

class Levels(Enum):
    POTATO = Level("Potato", LevelType.SHORT, 8, buildLevel_Potato)