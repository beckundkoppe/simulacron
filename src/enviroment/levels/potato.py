from enviroment.blueprints.blueprints import (
    PaperBox,
    Table,
    WoodenCabinet,
    WoodenShelf,
    WoodenTable,
)
from enviroment.entity import AdvancedContainerEntity, AgentEntity, ContainerEntity, Entity, connect_rooms_with_door
from enviroment.levels.data import LevelSpec
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World


vague_instruction_str = "Place a potato on the table."
detailed_instruction_str = "Find the potato in the other room, take it and place it on the table."


def build_easy(detailed_instruction: bool) -> LevelSpec:
    """
    Easy Level, where the agent has to get the potato from the other room, and put it on the table.
    clear instruction.
    no object descriptions.
    """

    main = Room("main", 4, 4)
    chamber = Room.chamber()

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    key.enter(main)

    table = Table(pos=Position(0.0, 3.0))
    table.enter(main)

    connect_rooms_with_door(main, Position(4.0, 4.0), chamber, Position(0.0, 1.0))

    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    chest = ContainerEntity(name="chest")
    chest.add_child(potato)

    shelf1 = ContainerEntity("shelf", pos=Position(0.0, 0.0))
    shelf1.enter(chamber)
    shelf1.add_child(chest)

    shelf2 = ContainerEntity("box", pos=Position(0.0, 2.0))
    shelf2.enter(chamber)
    shelf2.add_child(diamond)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_str if detailed_instruction else vague_instruction_str
        )],
        is_success=check_win_table_key,
    )


def build_medium(detailed_instruction: bool) -> LevelSpec:
    """
    Medium Level:
    The agent must find the correct container holding the potato. The container needs to be opened.
    - Multiple distractor containers and objects.
    - No locks, but the chest needs to be opened.
    """
    main = Room("main", 4, 4)
    chamber = Room.chamber()

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    # Main room:
    table = Table(pos=Position(0.0, 4.0))
    table.enter(main)

    diamond_main = Entity("diamond", Position(0.0, 0.0), is_collectible=True)
    chest_fake = ContainerEntity(name="box", pos=Position(1.0, 1.0))
    chest_fake.enter(main)
    chest_fake.add_child(diamond_main)

    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    key.enter(main)

    connect_rooms_with_door(main, Position(4.0, 4.0), chamber, Position(0.0, 1.0))

    # Chamber:
    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    shelf_fake = ContainerEntity("box", pos=Position(0.0, 0.0))
    shelf_fake.enter(chamber)
    shelf_fake.add_child(diamond)

    chest = AdvancedContainerEntity("chest", is_locked=True, is_open=False)
    chest.register_key(key)
    chest.add_child(potato)

    shelf_real = ContainerEntity("shelf", pos=Position(0.0, 2.0))
    shelf_real.enter(chamber)
    shelf_real.add_child(chest)

    return LevelSpec(
        agent_entities=[(
            tron,
            detailed_instruction_str if detailed_instruction else vague_instruction_str
        )],
        is_success=check_win_table_key,
    )


def build_hard(detailed_instruction: bool) -> LevelSpec:
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

    tron = AgentEntity("tron", pos=Position(0.0, 0.0))
    tron.enter(main)

    # Main room:
    key = Entity("key", Position(3.0, 1.0), is_collectible=True)
    table = WoodenTable(pos=Position(0.0, 4.0))
    table.enter(main)
    table.add_child(key)

    diamond_main = Entity("diamond", Position(0.0, 0.0), is_collectible=True)
    chest_fake = WoodenCabinet(pos=Position(1.0, 1.0))
    chest_fake.enter(main)
    chest_fake.add_child(diamond_main)

    connect_rooms_with_door(main, Position(4.0, 4.0), chamber, Position(0.0, 1.0))

    # Chamber:
    potato = Entity("potato", is_collectible=True)
    diamond = Entity("diamond", is_collectible=True)

    shelf_fake = PaperBox(pos=Position(0.0, 0.0))
    shelf_fake.enter(chamber)
    shelf_fake.add_child(diamond)

    chest = AdvancedContainerEntity("chest", is_locked=True, is_open=False)
    chest.register_key(key)
    chest.add_child(potato)

    shelf_real = WoodenShelf(pos=Position(0.0, 2.0))
    shelf_real.enter(chamber)
    shelf_real.add_child(chest)

    instr = (
        "Retrieve the potato locked in the chest in the other room. Use the key on the table "
        "to unlock it, take the potato, and place it on the table."
        if detailed_instruction
        else "Find the potato and place it on the table."
    )

    return LevelSpec(
        agent_entities=[(
            tron,
            instr
        )],
        is_success=check_win_table_key,
    )


def check_win_table_key() -> bool:
    potato = None
    table: ContainerEntity | None = None

    for ent in World.entities:
        if ent.name == "potato":
            potato = ent
        if ent.name == "table":
            table = ent

    assert potato is not None, "No potato"
    assert table is not None, "No table"

    return potato in table.children
