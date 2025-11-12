from enum import Enum, auto
from enviroment.action import ActionTry, ActionType
from agent.helper import trycatch, check_id
import current
from enviroment.entity import AgentEntity
from enviroment.exception import HardException
from enviroment.position import Position
from llm.tool import tool
from llm.toolprovider import ToolProvider

def _get_agent() -> AgentEntity:
    agent: AgentEntity = current.ENTITY

    if agent is None:
        raise Exception("No agent entity is available to perform the movement.")
    
    return agent

@tool
def move_to_position(x: str, y: str) -> str:
    """The human moves to a position.

    Args:
        x (float): the x position
        y (float): the y position
    """

    def _perform_move():
        agent = _get_agent()
        room = agent.room
        
        try:
            position = Position.from_input(x, y, room)
        except ValueError as exc:
            raise HardException(
                "The provided coordinates could not be interpreted.",
                console_message=(
                    f"Failed to parse coordinates x='{x}', y='{y}'"
                ),
                hint="Consult the observation field 'position_format' for the expected coordinate style.",
            )
    
        return agent.move_to_position(position)

    trycatch(_perform_move, "moved succesfully")

    return ""

@tool
def move_to_object(object_id: str) -> str:
    """The human moves to a position.

    Args:
        object_id (str): the id of the object to move to
    """

    agent = _get_agent()

    trycatch(lambda: agent.move_to_object(check_id(object_id)), "moved succesfully")

    return ""
        
@tool
def take_from(what_id: str, from_id: str) -> str:
    """The human picks up an item A from a object B and adds it to their inventory. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the id of item A to be taken.
        from_id (str): the id of object B from which the item is taken or 'FLOOR' for the floor
    """

    agent: AgentEntity = current.ENTITY
    
    if(from_id.upper() == "FLOOR"):
        trycatch(lambda: agent.take(check_id(what_id)), f"collected {what_id}")

    else:
        trycatch(lambda: agent.take_from(check_id(what_id), check_id(from_id)), f"collected {what_id} from {from_id}")

    return ""

@tool
def drop_to(what_id: str, to_id: str) -> str:
    """The human drops an item A from their inventory into or onto another object B. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the the id of the item A to be dropped.
        to_id (str): the id of the object B where the item is placed or 'FLOOR' for the floor
    """

    agent: AgentEntity = current.ENTITY

    if(to_id.upper() == "FLOOR"):
        trycatch(lambda: agent.drop(check_id(what_id)), f"dropped {what_id}")
    else:
        trycatch(lambda: agent.drop_into(check_id(what_id), check_id(to_id)), f"dropped {what_id} into {to_id}")

    return ""

@tool
def interact_with_object(object_id: str, operator: str) -> str:
    """The agent interacts with an object using a specific operator.
    
    Args:
        object_id (str): the the id of the item A to be dropped.
        operator (str): The action to perform. Allowed values: OPEN, CLOSE, GO_THROUGH.
    """

    agent: AgentEntity = current.ENTITY

    def helper():
        if(operator.upper() == "OPEN"):
            action = ActionTry(ActionType.OPEN)
        elif (operator.upper() == "CLOSE"):
            action = ActionTry(ActionType.CLOSE)
        elif (operator.upper() == "GO_THROUGH"):
            action = ActionTry(ActionType.USE)
        else:
            raise HardException("unknown operator for this action: {operator}")
        
        return check_id(object_id).on_interact(agent, action)
    
    trycatch(helper, f"succeded with {operator} {object_id}")

    return ""

@tool
def interact_with_object_using_item(object_id: str, using_id: str, operator: str) -> str:
    """The agent uses an item from inventory to interact with an object.

    Args:
        object_id (str): The id of the object to interact with.
        using_id (str): The id of the item to use from inventory.
        operator (str): The action to perform. Allowed values: LOCK, UNLOCK.
    """

    agent: AgentEntity = current.ENTITY

    def helper():
        if(operator.upper() == "LOCK"):
            action = ActionTry(ActionType.LOCK, check_id(using_id))
        elif (operator.upper() == "UNLOCK"):
            action = ActionTry(ActionType.UNLOCK, check_id(using_id))
        else:
            raise HardException("unknown operator for this action: {operator}")

        return check_id(object_id).on_interact(agent, action)

    trycatch(helper, f"succeded with {operator} {object_id}")

    return ""

class ToolGroup(Enum):
    NONE = auto()
    ALL = auto()
    ENV = auto()

_TOOLS_ENV = [
        move_to_position,
        move_to_object,
        take_from,
        drop_to,
        interact_with_object,
        interact_with_object_using_item,
    ]

def register_tools(toolprovider: ToolProvider, tools: ToolGroup):
    selection = None

    if tools is ToolGroup.ENV:
        selection = _TOOLS_ENV

    toolprovider.register_tools(tools=selection)
    