from dataclasses import dataclass
from enum import Enum

from enviroment.entity import ContainerEntity, ConnectorEntity, Entity
from enviroment.room import Position, Room
from enviroment.world import World


class LevelType(Enum):
    SHORT           = "short horizon"
    LONG            = "long horizon"

@dataclass
class LevelSpec:
    name: str
    kind: LevelType
    

def buildLevel_Potato(actor: Entity = None):
    main = Room("main", 4, 4)
    chamber = Room.chamber()

    actor.enter(main)

    key = Entity("key", Position(3.0, 1.0))
    key.enter(main)
    
    table = Entity("table", Position(0.0, 3.0))
    table.enter(main)
    
    doorMain = ConnectorEntity("door", Position(3.0,3.0))
    doorMain.enter(main)

    doorChamber = ConnectorEntity("door", Position(0.0,1.0))
    doorChamber.enter(chamber)


    doorMain.connect(doorChamber)
    doorChamber.connect(doorMain)

    potato = Entity("potato")
    diamond = Entity("diamond")

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

#x - - -
#- - - k
#- - - -
#t - - d




class Level(Enum):
    class Short(Enum):
        POTATO = LevelSpec(
                    name="Find potato",
                    kind=LevelType.SHORT,
                )
        
        EXIT_SMALL = LevelSpec(
                    name="Exit small building",
                    kind=LevelType.SHORT,
                )
    class Long(Enum):
        EXIT_BIG = LevelSpec(
                    name="Exit big building",
                    kind=LevelType.LONG,
                )