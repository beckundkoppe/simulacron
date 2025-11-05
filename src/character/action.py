from __future__ import annotations
from enum import Enum, auto
import inspect

from advanced.tool import tool
from debug import console
from enviroment.entity_controller import EntityController
from enviroment.position import Position
from enviroment.world import World


CONTROL: EntityController | None = None

class ActionType(Enum):
    TAKE    = "take",
    PUT     = "put",
    OPEN    = "open",
    CLOSE   = "close",
    USE     = "use",

@tool
def move_to_position(x: str, y: str) -> str:
    """The human moves to a position.

    Args:
        x (float): the x coordinate
        y (float): the y coordinate
    """
    assert CONTROL is not None, "set action.CONTROL"

    pos = Position(int(x), int(y))

    CONTROL.move_to_position(pos)

    console.pretty(
        console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: ({x},{y})", color=console.Color.YELLOW),
    )

    return ""

@tool
def use_door(door_id: str) -> str:
    """The human goes through a door.
    
    Args:
        door_id (str): the door_id to go through
    """

    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == door_id):
            CONTROL.use_connector(entity.uuid)

            console.pretty(
                console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {door_id}", color=console.Color.YELLOW),
            )

            return ""
