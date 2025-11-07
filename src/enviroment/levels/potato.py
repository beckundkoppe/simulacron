from enviroment.blueprints.blueprints import Chest, IronChest, PaperBox, Shelf, Table, WoodenCabinet, WoodenChest, WoodenShelf, WoodenTable
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.interaction import ObserverPerception
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World


def build_easy() -> LevelSpec:
    """
    Easy Level, where the agent has to get the potato from the other room, and put it on the table. 
    clear instruction.
    no object descriptions.
    """

    main = Room("main", 4, 4)
    chamber = Room.chamber()


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


    shelf1 = ContainerEntity("shelf", pos=Position(0.0,0.0))
    shelf1.enter(chamber)
    shelf1.add_child(chest)

    shelf2 = ContainerEntity("box",pos=Position(0.0,2.0))
    shelf2.enter(chamber)
    shelf2.add_child(diamond)

    return LevelSpec(
        agent_entities=[
            (tron,"Goal: Find the potato in the other room, take it and place it on the table. GIVE the next toolcall - nothing more"),
        ],
        is_success=check_win_table_key,
    )

def build_medium() -> LevelSpec:
    """
    Medium Level:
    The agent must find the correct container holding the potato. The container needs to be opened.
    - Multiple distractor containers and objects.
    - No locks, but the chest needs to be opened.
    """
    main = Room("main", 4, 4)
    chamber = Room.chamber()

    global MAIN
    global CHAMBER
    MAIN = main
    CHAMBER = chamber

    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    tron.enter(main)

    # Main room:

    table = Table(pos=Position(0.0, 4.0))
    table.enter(main)

    diamond_main = Entity("diamond", Position(0.0, 0.0), is_collectible=True)
    chest_fake = AdvancedContainerEntity(name="box", pos=Position(1.0,1.0),is_open=False, is_locked=False)
    chest_fake.enter(main)
    chest_fake.add_child(diamond_main)

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    key.enter(main)


    door_main = ConnectorEntity("door", Position(4.0, 4.0))
    door_main.enter(main)

    ##### Chamber:

    door_chamber = ConnectorEntity("door", Position(0.0, 1.0))
    door_chamber.enter(chamber)
    

    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    shelf_fake = ContainerEntity("box", pos=Position(0.0, 0.0))
    shelf_fake.enter(chamber)
    shelf_fake.add_child(diamond)

    chest = AdvancedContainerEntity("chest", is_locked=False, is_open=False)
    chest.add_child(potato)

    shelf_real = ContainerEntity("shelf", pos=Position(0.0, 2.0))
    shelf_real.enter(chamber)
    shelf_real.add_child(chest)

    door_main.connect(door_chamber)
    door_chamber.connect(door_main)

    return LevelSpec(
        agent_entities=[(
            tron,
            "Goal: Search the containers in the other room, find the potato, take it, and place it on the table. GIVE the next toolcall - nothing more."
        )],
        is_success=check_win_table_key,
    )

def build_hard() -> LevelSpec:
    """
    Hard Level:
    The potato is locked in a chest in the other room.
    The key is on the table in the main room.
    - Descriptive objects
    - similar/distracting objects
    - Multi-Step Reasoning
    """
    main = Room("main", 4, 4)
    chamber = Room.chamber()
    
    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    tron.enter(main)

    # Main room:

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    table = WoodenTable(pos=Position(0.0, 4.0))
    table.enter(main)
    table.add_child(key)

    diamond_main = Entity("diamond", Position(0.0, 0.0), is_collectible=True)
    chest_fake = AdvancedContainerEntity(name="shelf", pos=Position(1.0,1.0),is_open=False, is_locked=False, material="pine wood", description="An open shelf of rough pine planks, filled with dust and faint smell of resin.")
    chest_fake.enter(main)
    chest_fake.add_child(diamond_main)



    door_main = ConnectorEntity("door", Position(4.0, 4.0))
    door_main.enter(main)

    ##### Chamber:

    door_chamber = ConnectorEntity("door", Position(0.0, 1.0))
    door_chamber.enter(chamber)
    

    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    shelf_fake = PaperBox(pos=Position(0.0, 0.0))
    shelf_fake.enter(chamber)
    shelf_fake.add_child(diamond)

    chest = AdvancedContainerEntity("chest", is_locked=True, is_open=False, material="forged iron", description="A compact chest of black iron, its hinges groaning faintly when moved.")
    chest.add_key(key)
    chest.add_child(potato)

    shelf_real = WoodenShelf(pos=Position(0.0, 2.0))
    shelf_real.enter(chamber)
    shelf_real.add_child(chest)

    door_main.connect(door_chamber)
    door_chamber.connect(door_main)

    return LevelSpec(
        agent_entities=[(
            tron,
            "Goal: Retrieve the potato locked in the chest in the other room. Use the key on the table to unlock it, take the potato, and place it on the table. GIVE the next toolcall - nothing more.",
        )],
        is_success=check_win_table_key,
    )

def check_win_table_key() -> bool:
    potato = None
    table: ContainerEntity = None

    for ent in World.entities:
        if(ent.name == "potato"):
            potato = ent
        if(ent.name == "table"):
            table = ent
    
    assert potato != None, "No potato"
    assert table != None, "No table"
    
    return potato in table.children