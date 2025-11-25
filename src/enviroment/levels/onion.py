from enviroment.blueprints.blueprints import Box, Table
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity, connect_rooms_with_door
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World

detailed_instruction_str = "Bring the onion into the compost outside. From the living room got into the hallway and from there into the mudroom, through the main door into the garden, where the compost is."
vague_instruction_str = "Bring the onion into the compost outside."

def build_easy(detailed_instruction: bool) -> LevelSpec:
    #----- living room -----
    living_room = Room("living_room", 4, 5)

    table = Table(Position(3, 3.25))
    table.enter(living_room)

    onion = Entity("onion", Position(0.0, 0.0), is_collectible=True)
    table.add_child(onion)

    #----- chamber -----
    chamber = Room("chamber", 1, 1.5)


    #----- hallway -----
    hallway = Room("hallway", 1, 5.5)


    #----- mudroom -----
    mudroom = Room("mudroom", 2, 2)


    #----- garden -----
    garden = Room("garden", 5, 10)

    compost = ContainerEntity("compost box", Position(0.5, 4.25))
    compost.enter(garden)


    #----- kitchen -----
    kitchen = Room("kitchen", 4, 3)


    #----- office -----
    office = Room("office", 3, 3)


    #----- storage room -----
    storage_room = Room("storage_room", 4, 3)


    #----- bed room -----
    bed_room = Room("bed_room", 2, 2)


    #----- bath -----
    bath = Room("bath", 2, 3)


    ## livingroom <-> kitchen
    connect_rooms_with_door(living_room, Position(0.2, 5), kitchen, Position(0.2, 0))

    # livingroom <-> hallway
    connect_rooms_with_door(living_room, Position(4,1.75), hallway, Position(0,1.75))

    # storage_room <-> kitchen
    connect_rooms_with_door(storage_room, Position(3.5,0), kitchen, Position(3.5,3))

    # kitchen <-> hallway
    connect_rooms_with_door(kitchen, Position(4,1), hallway, Position(0,4.5))

    # hallway <-> office
    connect_rooms_with_door(hallway,  Position(0.5, 5.5), office, Position(0.5,0))

    # mudroom <-> hallway
    connect_rooms_with_door(mudroom, Position(0,1), hallway, Position(1,4.5))

    # mudroom <-> garden
    connect_rooms_with_door(mudroom, Position(2,1), garden, Position(0,6))

    # hallway <-> bed_room
    connect_rooms_with_door(hallway,  Position(1, 2.5), bed_room, Position(0,1))

    # bed_room <-> bath
    connect_rooms_with_door(bed_room,  Position(1.5, 0), bath, Position(1.5,3))

    # hallway <-> chamber
    connect_rooms_with_door(hallway,  Position(0.5,0), chamber, Position(0.5, 1.5))

    tron = AgentEntity("tron", pos=Position(0.75, 0.75))
    tron.enter(living_room)

    return LevelSpec(
        agent_entities=[(tron,detailed_instruction_str if detailed_instruction else vague_instruction_str),],
        is_success=check_win,
    )

def build_hard(detailed_instruction) -> LevelSpec:
    #----- Ablenkung -----

    shelf_1 = ContainerEntity("shelf", pos=Position(0, 2.5))
    shelf_1.enter(office)

    apple_1 = Entity("apple", Position(0.0, 0.0), is_collectible=True)
    apple_2 = Entity("apple", Position(0.0, 0.0), is_collectible=True)
    shelf_1.add_child(apple_1)
    shelf_1.add_child(apple_2)

    chair_1 = ContainerEntity("chair", pos=Position(1.0, 3.5))
    chair_1.enter(living_room)

    chair_2 = ContainerEntity("chair", pos=Position(0.0, 4.5))
    chair_2.enter(living_room)

    return LevelSpec(
        agent_entities=[(tron,detailed_instruction_str if detailed_instruction else vague_instruction_str),],
        is_success=check_win,
    )

def check_win() -> bool:
    onion = None
    compost_box: ContainerEntity = None

    for ent in World.entities:
        if(ent.name == "onion"):
            onion = ent
        if(ent.name == "compost box"):
            compost_box = ent
    
    assert onion != None, "No onion"
    assert compost_box != None, "No compost box"
    
    return onion in compost_box.children