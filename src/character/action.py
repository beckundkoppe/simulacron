from __future__ import annotations
from enum import Enum, auto
import inspect

from advanced.tool import tool
from debug import console
from enviroment.position import Position
from enviroment.world import World


CONTROL = None

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

@tool
def take_from_room(item_id: str) -> str:
    """The human picks up an item from the room into his inventory.
    
    Args:
        item_id (str): the id of the object
    """

    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == item_id):
            CONTROL.take(entity.uuid)

            console.pretty(
                console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {item_id}", color=console.Color.YELLOW),
            )

            return ""


@tool
def drop_into_room(item_id: str) -> str:
    """The human drops an item from his inventory into the room.
    
    Args:
        item_id (str): the the id of the object
    """

    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == item_id):
            CONTROL.drop(entity.uuid)

            console.pretty(
                console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {item_id}", color=console.Color.YELLOW),
            )

            return ""
        
@tool
def take_from(item_id: str, from_id: str) -> str:
    """The human picks up an item A from a object B and adds it to their inventory. Only use this, when you are nearby B.
    
    Args:
        item_id (str): the id of item A to be taken.
        from_id (str): the id of object B from which the item is taken.
    """

    item_ent = None
    from_ent = None

    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == item_id):
            item_ent = entity
        if(entity.readable_id == from_id):
            from_ent = entity

    if((from_ent is not None) and item_ent is not None):
        CONTROL.take_from(item_ent.uuid, from_ent.uuid)
        console.pretty(
            console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {item_id} from {from_id}", color=console.Color.YELLOW),
        )

    return ""


@tool
def drop_to(item_id: str, to_id: str) -> str:
    """The human drops an item A from their inventory into or onto another object B. Only use this, when you are nearby B.
    
    Args:
        item_id (str): the the id of the item A to be dropped.
        to_id (str): the id of the object B where the item is placed.
    """

    item_ent = None
    from_ent = None

    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == item_id):
            item_ent = entity
        if(entity.readable_id == to_id):
            from_ent = entity

    if((from_ent is not None) and item_ent is not None):
        CONTROL.drop_into(item_ent.uuid, from_ent.uuid)
        console.pretty(
            console.bullet(f"[TOOLCALL] {inspect.currentframe().f_code.co_name}: {item_id} to {to_id}", color=console.Color.YELLOW),
        )

    return ""
        

TOOLS = [
    move_to_position, use_door,
    take_from, drop_to,
]
