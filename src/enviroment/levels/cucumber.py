from enviroment.blueprints.blueprints import Chest, Table, WoodenTable
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World

vague_instruction_str = "Place the cucumber on the table."
detailed_instruction_str = "Find the cucumber, pick it up and place it on the table."

def build_easy(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 2, 4)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    key = Entity("cucumber", Position(3.5, 1.5), is_collectible=True)
    key.enter(main)
    
    table = Table(pos=Position(3.5,0.5))
    table.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_str if detailed_instruction else vague_instruction_str
            ),
        ],
        is_success=check_win,
    )

def build_hard(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 2, 4)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    key = Entity("cucumber", Position(3.5, 1.5), is_collectible=True)
    key.enter(main)
    
    table = WoodenTable(pos=Position(3.5,0.5))
    table.enter(main)

    # geschlossene Kiste bei (2,2)
    chest = Chest(pos=Position(2.0, 2.0), description="A wooden chest.")
    chest.is_open = False
    chest.enter(main)

    # potato liegt frei bei (0.5, 1.5)
    potato = Entity("potato", Position(0.5, 1.5), is_collectible=True)
    potato.enter(main)

    instr = ""
    if detailed_instruction:
        instr = "Find the cucumber, pick it up and place it on the table"
    else:
        instr = "Place the cucumber on the table."

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_str if detailed_instruction else vague_instruction_str
        )],
        is_success=check_win,
    )


def check_win() -> bool:
    cucumber = None
    table: ContainerEntity = None

    for ent in World.entities:
        if(ent.name == "cucumber"):
            cucumber = ent
        if(ent.name == "table"):
            table = ent
    
    assert cucumber != None, "No potato"
    assert table != None, "No table"
    
    return cucumber in table.children