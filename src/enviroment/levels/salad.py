from enviroment.blueprints.blueprints import Box, Chest, MetalShelf, Shelf, Table
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ConnectorEntity, ContainerEntity, Entity
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World


vague_instruction_salad = (
    "Sort the salad types: put spinach into the box and lettuce into the shelf."
)
detailed_instruction_salad = (
    "There are three spinach and three lettuce objects in the room. "
    "Spinach must go into the box, and lettuce must be placed in the shelf. "
    "For each item: move to it, pick it up, move to the correct target and place it."
)


def build_easy(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 5, 5)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    # Ablageorte
    chest = Box(pos=Position(4.0, 1.0))
    chest.enter(main)

    shelf = Shelf(pos=Position(4.0, 3.0))
    shelf.enter(main)

    # 6 Positionen für die 6 Objekte
    salad_positions = [
        Position(1.0, 1.0),
        Position(2.0, 1.0),
        Position(3.0, 1.0),
        Position(1.0, 3.0),
        Position(2.0, 3.0),
        Position(3.0, 3.0),
    ]

    # 3x spinach
    for i in range(3):
        e = Entity("spinach", salad_positions[i], is_collectible=True)
        e.enter(main)

    # 3x lettuce
    for i in range(3):
        e = Entity("lettuce", salad_positions[3 + i], is_collectible=True)
        e.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_salad if detailed_instruction else vague_instruction_salad
        )],
        is_success=check_salad_win,
    )

def build_heard(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 5, 5)

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    # Ablageorte
    chest = Box(pos=Position(0.0, 5.0), description="A paper box")
    chest.enter(main)

    shelf = MetalShelf(pos=Position(5.0, 5.0))
    shelf.enter(main)

    # 6 Positionen für die 6 Objekte
    salad_positions = [
    Position(1.5, 3.0),
    Position(2.0, 0.5),
    Position(3.5, 4.5),
    Position(0.5, 2.5),
    Position(2.5, 3.5),
    Position(5.0, 1.5),
]
    # 3x spinach
    for i in range(3):
        e = Entity("spinach", salad_positions[i], is_collectible=True)
        e.enter(main)

    # 3x lettuce
    for i in range(3):
        e = Entity("lettuce", salad_positions[3 + i], is_collectible=True)
        e.enter(main)

    # Ablenkungs-Gemüse
    distraction_positions = [
        Position(1.0, 1.0),
        Position(4.5, 0.5),
        Position(3.0, 2.0),
        Position(0.5, 4.0),
        #Position(2.5, 1.5),
        #Position(4.0, 3.5),
    ]

    distraction_names = [
        "carrot",
        "tomato",
        "onion",
        "pepper",
        #"broccoli",
        #"cucumber",
    ]

    for name, pos in zip(distraction_names, distraction_positions):
        e = Entity(name, pos, is_collectible=True)
        e.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_salad if detailed_instruction else vague_instruction_salad
        )],
        is_success=check_salad_win,
    )



def check_salad_win() -> bool:
    box = None
    shelf = None

    spinach = []
    lettuce = []

    for ent in World.entities:
        if ent.name == "spinach":
            spinach.append(ent)
        elif ent.name == "lettuce":
            lettuce.append(ent)
        elif ent.name == "box":
            box = ent
        elif ent.name == "shelf":
            shelf = ent

    assert box is not None, "box missing"
    assert shelf is not None, "Shelf missing"
    

    # Bedingungen:
    # Spinach → chest
    # Lettuce → shelf
    return (
        all(item in box.children for item in spinach) and
        all(item in shelf.children for item in lettuce)
    )