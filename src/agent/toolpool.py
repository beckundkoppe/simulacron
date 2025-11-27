from enum import Enum, auto
from typing import List
from enviroment.action import ActionTry, ActionType
from agent.helper import trycatch, check_id
import current
from enviroment.entity import AgentEntity
from enviroment.exception import HardException
from enviroment.position import Position
from llm.tool import tool
from agent.plan import TreePlan
from llm.toolprovider import ToolProvider

@tool
def move_to_position(x: str, y: str) -> str:
    """The human moves to a position.

    Args:
        x (float): the x position
        y (float): the y position
    """

    entity = current.ENTITY
    room = entity.room
    old_x = entity.pos.x
    old_y = entity.pos.y

    def _perform_move():
        try:
            position = Position.from_input(x, y)
        except ValueError as exc:
            raise HardException(
                "The provided coordinates could not be interpreted.",
                console_message=(
                    f"Failed to parse coordinates x='{x}', y='{y}'"
                ),
                hint="Consult the observation field 'position_format' for the expected coordinate style.",
            )
    
        return entity.move_to_position(position)

    trycatch(_perform_move, f"moved from ({old_x},{old_y}) to ({x},{y})")

    return ""

@tool
def move_to_object(object_id: str) -> str:
    """The human moves to a position.

    Args:
        object_id (str): the id of the object to move to
    """

    entity = current.ENTITY
    old_x = entity.pos.x
    old_y = entity.pos.y

    trycatch(lambda: entity.move_to_object(check_id(object_id)), f"moved from ({old_x},{old_y}) to {object_id}")

    return ""
        
@tool
def take_from(what_id: str, from_id: str) -> str:
    """The human picks up an item A from a object B and adds it to their inventory. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the id of item A to be taken.
        from_id (str): the id of object B from which the item is taken or 'FLOOR' for the floor
    """

    entity: AgentEntity = current.ENTITY
    
    if(from_id.upper() == "FLOOR"):
        trycatch(lambda: entity.take(check_id(what_id)), f"collected {what_id}")

    else:
        trycatch(lambda: entity.take_from(check_id(what_id), check_id(from_id)), f"collected {what_id} from {from_id}")

    return ""

@tool
def drop_to(what_id: str, to_id: str) -> str:
    """The human drops an item A from their inventory into or onto another object B. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the the id of the item A to be dropped.
        to_id (str): the id of the object B where the item is placed or 'FLOOR' for the floor
    """

    entity: AgentEntity = current.ENTITY

    if(to_id.upper() == "FLOOR"):
        trycatch(lambda: entity.drop(check_id(what_id)), f"dropped {what_id}")
    else:
        trycatch(lambda: entity.drop_into(check_id(what_id), check_id(to_id)), f"dropped {what_id} into {to_id}")

    return ""

@tool
def interact_with_object(object_id: str, operator: str) -> str:
    """The human interacts with an object using a specific operator.

    Args:
        object_id (str): the the id of the item A to be dropped.
        operator (str): The action to perform. Allowed values: OPEN, CLOSE, GO_THROUGH, LOOK_THROUGH.
    """

    entity: AgentEntity = current.ENTITY

    def helper():
        if(operator.upper() == "OPEN"):
            action = ActionTry(ActionType.OPEN)
        elif (operator.upper() == "CLOSE"):
            action = ActionTry(ActionType.CLOSE)
        elif (operator.upper() == "GO_THROUGH"):
            action = ActionTry(ActionType.USE)
        elif (operator.upper() == "LOOK_THROUGH"):
            action = ActionTry(ActionType.LOOK_THROUGH)
        else:
            raise HardException(f"unknown operator for this action: can not {operator} {object_id}")
        
        return check_id(object_id).on_interact(entity, action)
    
    trycatch(helper, f"succeded with {operator} {object_id}")

    return ""

@tool
def interact_with_object_using_item(object_id: str, using_id: str, operator: str) -> str:
    """The human uses an item from inventory to interact with an object.

    Args:
        object_id (str): The id of the object to interact with.
        using_id (str): The id of the item to use from inventory.
        operator (str): The action to perform. Allowed values: LOCK, UNLOCK.
    """

    entity: AgentEntity = current.ENTITY

    def helper():
        if(operator.upper() == "LOCK"):
            action = ActionTry(ActionType.LOCK, check_id(using_id))
        elif (operator.upper() == "UNLOCK"):
            action = ActionTry(ActionType.UNLOCK, check_id(using_id))
        else:
            raise HardException(f"unknown operator for this action: can not {operator} {object_id}")

        return check_id(object_id).on_interact(entity, action)

    trycatch(helper, f"succeded with {operator} {object_id}")

    return ""

@tool
def add_step(step: str) -> str:
    """Add the next plan step

    Args:
        step (str): A short description of the next step.
    """

    agent = current.AGENT

    if agent and hasattr(agent, "plan") and hasattr(agent.plan, "add_step"):
        agent.plan.add_step(step)
    else:
        raise ValueError("Current agent plan does not support adding steps")

    return ""

@tool
def add_trial(trial: str) -> str:
    """Add the next plan step

    Args:
        trial (str): A short description of an approach to fullfill the plan.
    """

    agent = current.AGENT

    if agent and hasattr(agent, "plan") and hasattr(agent.plan, "trial_ideas"):
        agent.plan.trial_ideas.append(trial)
    else:
        raise ValueError("Current agent plan does not support adding trials")

    return ""

@tool
def decompose_node(task_node_id: int, sub_nodes: list[str]) -> str:
    """Decompose a task node into a list of subtasks

    Args:
        task_node_id (int): The ID of the task node to decompose
        sub_nodes (list[str]): The list of the sub task with a short description each
    """

    agent = current.AGENT

    if agent and isinstance(agent.plan, TreePlan):
        agent.plan.decompose_node(task_node_id, sub_nodes)
    else:
        raise ValueError("Current agent plan does not support decomposition")

    return ""

@tool
def delete_node(task_node_id: int, delete_children:bool = True) -> str:
    """Deletes a task node

    Args:
        task_node_id (int): The ID of the task node to delete
        delete_children (bool): Wether children should be deleted or moved up
    """

    agent = current.AGENT

    if agent and isinstance(agent.plan, TreePlan):
        agent.plan.delete_node(task_node_id, delete_children)
    else:
        raise ValueError("Current agent plan does not support deleting nodes")

    return ""

@tool
def mark_done(task_node_id: int) -> str:
    """Marks a plan node as completed.

    Args:
        task_node_id (int): The ID of the task node to mark as done
    """

    agent = current.AGENT

    if agent and isinstance(agent.plan, TreePlan):
        agent.plan.mark_node_done(task_node_id)
    else:
        raise ValueError("Current agent plan does not support marking nodes done")

    return ""

@tool
def mark_focused(task_node_id: int) -> str:
    """Marks a plan node as the current focus.

    Args:
        task_node_id (int): The ID of the task node to focus
    """

    agent = current.AGENT

    if agent and isinstance(agent.plan, TreePlan):
        agent.plan.mark_node_focus(task_node_id)
    else:
        raise ValueError("Current agent plan does not support marking focus")

    return ""

@tool
def store_memory(information: str) -> str:
    """Store an information permanently. For relevant data only.

    Args:
        information (str): The compact information to store.
    """

    agent = current.AGENT

    memory_id = agent.main_memory.store_permanent_memory(information)

    return f"Stored memory with id {memory_id}"

@tool
def delete_memory(memory_id: int,) -> str:
    """Deletes a memory

    Args:
        memory_id (int): The ID of the memory to delete
    """

    agent = current.AGENT

    agent.main_memory.delete_permanent_memory(memory_id)

    return ""

@tool
def yes() -> str:
    """Answer with yes

    Args:
        None
    """

    current.ANSWER_BUFFER = True

    return ""

@tool
def no() -> str:
    """Answer with no

    Args:
        None
    """

    current.ANSWER_BUFFER = False

    return ""

@tool
def yes_rationale(rationale: str) -> str:
    """Answer with yes and the rationale

    Args:
        rationale (str): why yes
    """

    current.ANSWER_BUFFER = True
    current.ANSWER_BUFFER_REASON = rationale

    return ""

@tool
def no_rationale(rationale: str) -> str:
    """Answer with no and the rationale

    Args:
        rationale (str): why no
    """

    current.ANSWER_BUFFER = False
    current.ANSWER_BUFFER_REASON = rationale

    return ""

class ToolGroup(Enum):
    NONE    = auto()
    ALL     = auto()
    ENV     = auto()
    MEM     = auto()
    PLAN    = auto()
    DECOMPOSE    = auto()
    QA    = auto()
    QA_RATIO = auto()
    TRIAL = auto()

_TOOLS_ENV = [
        move_to_position,
        move_to_object,
        take_from,
        drop_to,
        interact_with_object,
        interact_with_object_using_item,
    ]

_TOOLS_MEM = [
        store_memory,
        delete_memory,
    ]

_TOOLS_PLAN = [
        add_step,
    ]

_TOOLS_TRIAL = [
        add_trial,
    ]

_TOOLS_DECOMPOSE = [
        decompose_node,
        delete_node,
        mark_done,
        mark_focused,
    ]

_TOOLS_QA = [
        yes,
        no,
    ]

_TOOLS_QA_RATIO = [
        yes_rationale,
        no_rationale,
    ]

def register_tools(toolprovider: ToolProvider, tools):
    if isinstance(tools, ToolGroup):
        tools = [tools]

    selection = []

    is_all = ToolGroup.ALL in tools

    if ToolGroup.ENV in tools or is_all:
        selection.extend(_TOOLS_ENV)

    if ToolGroup.MEM in tools or is_all:
        selection.extend(_TOOLS_MEM)

    if ToolGroup.PLAN in tools or is_all:
        selection.extend(_TOOLS_PLAN)

    if ToolGroup.DECOMPOSE in tools or is_all:
        selection.extend(_TOOLS_DECOMPOSE)

    if ToolGroup.QA in tools or is_all:
        selection.extend(_TOOLS_QA)
    
    if ToolGroup.QA_RATIO in tools or is_all:
        selection.extend(_TOOLS_QA_RATIO)

    if ToolGroup.TRIAL in tools or is_all:
        selection.extend(_TOOLS_TRIAL)

    toolprovider.register_tools(tools=selection)
    