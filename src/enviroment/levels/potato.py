from enviroment.blueprints.blueprints import Chest, PaperBox, Shelf, Table, WoodenCabinet, WoodenChest, WoodenShelf
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.interaction import ObserverPerception
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World

MAIN = None
CHAMBER = None

def build_easy() -> LevelSpec:
    main = Room("main", 4, 4)
    chamber = Room.chamber()

    global MAIN
    global CHAMBER
    MAIN = main
    CHAMBER = chamber

    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    tron.enter(main)

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    key.enter(main)
    
    table = Table(pos=Position(0.0,3.0))
    table.enter(main)
    
    doorMain = ConnectorEntity("door", Position(4.0,4.0))
    doorMain.enter(main)

    doorChamber = ConnectorEntity("door", Position(0.0,1.0))
    doorChamber.enter(chamber)

    doorMain.connect(doorChamber)
    doorChamber.connect(doorMain)

    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    chest = AdvancedContainerEntity(name="chest")
    chest.add_child(potato)

    box = PaperBox()
    box.add_child(diamond)

    shelf1 = WoodenShelf(pos=Position(0.0,0.0))
    shelf1.enter(chamber)
    shelf1.add_child(chest)

    shelf2 = WoodenShelf(pos=Position(0.0,2.0))
    shelf2.enter(chamber)
    shelf2.add_child(box)

    return LevelSpec(
        agent_entities=[
            (tron,"Goal: Find a potato, take it and place it on the table. GIVE the next toolcall - nothing more"),
        ],
        is_success=check_win,
    )

def build_hard() -> LevelSpec:
    spec = build_easy()

    chest = WoodenChest(pos=Position(2.0,4.0))
    chest.enter(MAIN)

    cabinet = WoodenCabinet(pos=Position(4.0,2.0))
    cabinet.enter(MAIN)

    return spec

def check_win() -> bool:
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