from enviroment.blueprints.blueprints import Cabinet, Chest, IronTable, PaperBox, Shelf, Table, Box, MetalShelf, WoodenChest, WoodenShelf
from enviroment.capabilities import OpenableCapability
from enviroment.entity import AgentEntity, ContainerEntity, Entity
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World


vague_instruction_tomato = (
    "Pack the tomato: put it into the box, then put the box into the crate, "
    "close everything and place the crate on the shelf."
)

detailed_instruction_tomato = (
    "A tomato lies on the table. A box, a crate and a shelf are in the room. "
    "First take the tomato from the table. Then put the tomato into the box and close the box. "
    "After that put the box into the crate and close the crate. "
    "Finally, place the closed crate on the shelf."
)


def build_easy(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 5, 5)

    # Agent
    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    tomato = Entity("tomato", pos=Position(1.0, 4.0), is_collectible=True)
    # Tisch mit Tomate
    table = Table(pos=Position(1.0, 4.0))
    table.enter(main)
    table.add_child(tomato)

    # Box (muss später geschlossen werden)
    box = Box(pos=Position(3.0, 3.0))
    box.is_collectible = True
    box._openable = OpenableCapability(
            box,
            initially_open=True,
        )
    box.add_capability(box._openable)
    box.enter(main)

    # Kiste (Crate)
    crate = Chest(pos=Position(4.0, 1.0))
    crate.is_collectible = True
    crate.is_open = True
    crate.enter(main)

    # Regal
    shelf = Shelf(pos=Position(2.0, 1.0))
    shelf.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_tomato if detailed_instruction else vague_instruction_tomato
        )],
        is_success=check_tomato_packaging_win,
    )

def build_hard(detailed_instruction: bool) -> LevelSpec:
    main = Room("main", 5, 5)

    # Agent
    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    tomato = Entity("tomato", pos=Position(1.0, 4.0), is_collectible=True)
    # Tisch mit Tomate
    table = IronTable(pos=Position(1.0, 4.0))
    table.enter(main)
    table.add_child(tomato)

    # Box (muss später geschlossen werden)
    box = PaperBox(pos=Position(3.0, 3.0))
    box.is_collectible = True
    box._openable = OpenableCapability(
            box,
            initially_open=True,
        )
    box.add_capability(box._openable)
    box.enter(main)

    # Kiste / Crate (Ziel 2 – echt)
    crate = WoodenChest(pos=Position(4.0, 1.0))
    crate.is_collectible = True
    crate.is_open = True
    crate.enter(main)

    # Regal (Ziel 3)
    shelf = WoodenShelf(pos=Position(2.0, 1.0))
    shelf.enter(main)

    # -------------------------------------------------------------
    # ⚠ Ablenkungen
    # -------------------------------------------------------------


    wrong_box = Cabinet(pos=Position(0.5, 1.5), description="A damaged cabinet.")
    wrong_box.is_open = False
    wrong_box.enter(main)

    wrong_crate = ContainerEntity(name="bin", pos=Position(3.5, 0.5), description="A trash bin.")
    wrong_crate.enter(main)

    # random Gemüse zur Ablenkung
    distraction_positions = [
        Position(4.5, 0.5),
        Position(0.5, 3.0),
        #Position(3.0, 2.0),
        #Position(1.5, 1.5),
    ]

    distraction_names = [
        "potato"
        "carrot",
        #"onion",
        #"pepper",
    ]

    for name, pos in zip(distraction_names, distraction_positions):
        e = Entity(name, pos, is_collectible=True)
        e.enter(main)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_tomato if detailed_instruction else vague_instruction_tomato
        )],
        is_success=check_tomato_packaging_win,
    )

def check_tomato_packaging_win() -> bool:
    tomato = None
    box = None
    chest = None
    shelf = None

    # Objekte im Raum finden
    for ent in World.entities:
        if ent.name == "tomato":
            tomato = ent
        elif ent.name == "box":
            box = ent
        elif ent.name == "chest":
            chest = ent
        elif ent.name == "shelf":
            shelf = ent

    assert tomato is not None, "Tomato missing"
    assert box is not None, "Box missing"
    assert chest is not None, "chest missing"
    assert shelf is not None, "Shelf missing"

    # Bedingungen:
    # 1. Tomate → in Box
    # 2. Box → in Kiste
    # 3. Kiste → im Regal
    # 4. Box geschlossen
    # 5. Kiste geschlossen

    return (
        tomato in box.children and
        box in chest.children and
        chest in shelf.children and
        not box._openable.is_open and
        not chest.is_open
    )
