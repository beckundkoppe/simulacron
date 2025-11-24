from enviroment.blueprints.blueprints import Table
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World

detailed_instruction_str = "Place the carrot on the table. Only one step is necessary."
vague_instruction_str = "Place the carrot on the table."

def build_easy(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 5, 5)

    tron = AgentEntity("tron", pos=Position(0.5, 0.5))
    tron.enter(main)
    
    table = Table(pos=Position(1.0,4.0))
    table.enter(main)

    carrot = Entity("carrot", is_collectible=True)
    tron.inventory.append(carrot)

    return LevelSpec(
        agent_entities=[(tron,detailed_instruction_str if detailed_instruction else vague_instruction_str),],
        is_success=check_win,
    )

def build_hard(detailed_instruction) -> LevelSpec:
    #----- Ablenkung -----

    shelf_1 = ContainerEntity("shelf", pos=Position(4.5, 4.5))
    shelf_1.enter(main)

    apple_1 = Entity("apple", Position(0.0, 0.0), is_collectible=True)
    apple_2 = Entity("apple", Position(0.0, 0.0), is_collectible=True)
    shelf_1.add_child(apple_1)
    shelf_1.add_child(apple_2)

    chair_1 = ContainerEntity("chair", pos=Position(1.0, 3.5))
    chair_1.enter(main)

    chair_2 = ContainerEntity("chair", pos=Position(0.0, 4.5))
    chair_2.enter(main)

    #----- Notwendig -----

    main = Room("main", 5, 5)

    tron = AgentEntity("tron", pos=Position(0.5, 0.5))
    tron.enter(main)
    
    table = Table(pos=Position(1.0,4.0))
    table.enter(main)

    carrot = Entity("carrot", is_collectible=True)
    tron.inventory.append(carrot)

    return LevelSpec(
        agent_entities=[(tron,detailed_instruction_str if detailed_instruction else vague_instruction_str),],
        is_success=check_win,
    )

def check_win() -> bool:
    carrot = None
    table: ContainerEntity = None

    for ent in World.entities:
        if(ent.name == "carrot"):
            carrot = ent
        if(ent.name == "table"):
            table = ent
    
    assert carrot != None, "No carrot"
    assert table != None, "No table"
    
    return carrot in table.children