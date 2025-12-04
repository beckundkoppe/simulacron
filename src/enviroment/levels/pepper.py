from enviroment.blueprints.blueprints import Chest, Table, WoodenTable
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World

vague_instruction_str = "Place all of the peppers on the table."
detailed_instruction_str = "Collect all Peppers, then place all of the peppers on the table."

def build_easy(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 4, 2)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    p1 = Entity("pepper", Position(3.5, 1.5), is_collectible=True)
    p1.enter(main)

    p2 = Entity("pepper", Position(3.0, 1.5), is_collectible=True)
    p2.enter(main)

    p3 = Entity("pepper", Position(3.5, 1.0), is_collectible=True)
    p3.enter(main)
    
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
    main = Room("main", 4, 2)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    p1 = Entity("pepper", Position(3.5, 1.5), is_collectible=True)
    p1.enter(main)

    p2 = Entity("pepper", Position(3.0, 1.5), is_collectible=True)
    p2.enter(main)

    p3 = Entity("pepper", Position(3.5, 1.0), is_collectible=True)
    p3.enter(main)
    
    table = WoodenTable(pos=Position(3.5,0.5))
    table.enter(main)

    # geschlossene Kiste bei (2,2)
    chest = Chest(pos=Position(2.0, 2.0), description="A wooden chest.")
    chest.is_open = False
    chest.enter(main)

    # potato liegt frei bei (0.5, 1.5)
    potato = Entity("potato", Position(0.5, 1.5), is_collectible=True)
    potato.enter(main)

    apple = Entity("apple", Position(3.5, 1.6), is_collectible=True)
    apple.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_str if detailed_instruction else vague_instruction_str
        )],
        is_success=check_win,
    )


def check_win() -> bool:
    peppers = []
    table: ContainerEntity = None

    for ent in World.entities:
        if(ent.name == "pepper"):
            peppers.append(ent)
        if(ent.name == "table"):
            table = ent
    
    assert len(peppers) != 3, "No peppers"
    assert table != None, "No table"
    
    for pep in peppers:
        if pep not in table.children:
            return False
        
    return True