from __future__ import annotations
from enum import Enum, auto
import inspect

from advanced.tool import tool
from character import current
from character.exception import HardException, SoftException
from character.resultbuffer import ActionNotPossible, FormalError, Success
from debug import console
from enviroment.position import Position
from enviroment.world import World

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

    trycatch(lambda: current.AGENT.entity.move_to_position(Position(float(x), float(y))), "moved succesfully")

    return ""

@tool
def use_door(door_id: str) -> str:
    """The human goes through a door.
    
    Args:
        door_id (str): the door_id to go through
    """
    trycatch(lambda: current.AGENT.entity.use_connector(check_id(door_id)), "went through door")

    return ""
        
@tool
def take_from(item_id: str, from_id: str) -> str:
    """The human picks up an item A from a object B and adds it to their inventory. Only use this, when you are nearby B.
    
    Args:
        item_id (str): the id of item A to be taken.
        from_id (str): the id of object B from which the item is taken or 'FLOOR' for the floor
    """
    
    if(from_id == "FLOOR"):
        trycatch(lambda: current.AGENT.entity.take(check_id(item_id)), f"collected {item_id}")

    else:
        trycatch(lambda: current.AGENT.entity.take_from(check_id(item_id), check_id(from_id)), f"collected {item_id} from {from_id}")

    return ""


@tool
def drop_to(item_id: str, to_id: str) -> str:
    """The human drops an item A from their inventory into or onto another object B. Only use this, when you are nearby B.
    
    Args:
        item_id (str): the the id of the item A to be dropped.
        to_id (str): the id of the object B where the item is placed or 'FLOOR' for the floor
    """

    if(to_id == "FLOOR"):
        trycatch(lambda: current.AGENT.entity.drop(check_id(item_id)), f"dropped {item_id}")
    else:
        trycatch(lambda: current.AGENT.entity.drop_into(check_id(item_id), check_id(to_id)), f"dropped {item_id} into {to_id}")

    return ""
        

def trycatch(action, success_message):
    try:
        action()
        Success(success_message)
    except SoftException as s:
        ActionNotPossible(str(s))
    except HardException as h:
        FormalError(str(h))

def check_id(readable_id: str) -> str:
    uuid = None
    for ent in World.entities:
        entity = World.get_entity(ent)
        if(entity.readable_id == readable_id):
            uuid = ent

    if(uuid != None):
        return uuid

    raise HardException(f"no such object '{readable_id}'")

TOOLS = [
    move_to_position, use_door,
    take_from, drop_to,
]
