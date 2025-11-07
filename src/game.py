import json
from advanced.agent import Agent
from advanced.tool import tool
from enviroment import current
from enviroment.action import ActionTry, ActionType
from enviroment.exception import HardException, SoftException
from enviroment.levels.level import Level, LevelSpec
from enviroment.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from debug import console
from enviroment.entity import AgentEntity
from enviroment.interaction import Depth, ObserverPerception
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World
from llm.memory.memory import Memory, Role

def observe(room: Room, observer: AgentEntity) -> str:
    data = {}
    data["you_are_in_room"] = {
        "name": room.name,
        "your_pos": {
            "x": observer.pos.x,
            "y": observer.pos.y,
        },
        "room_size": {
            "extend_x": room.extend_x,
            "extend_y": room.extend_y,
        }
    }
    data["your_inventory"] = observer.get_inventory()
    data["your_observation"] = room.perceive(observer, Depth.OMNISCIENT)
    return json.dumps(data)

def trycatch(action, success_msg):
    try:
        msg = ""
        message = action() #message is None <--------
        if message is None:
            msg = success_msg
        else:
            msg = message

        Success(msg)
        current.RESULT.toolcall_count += 1
    except SoftException as s:
        agent_msg = getattr(s, "agent_message", str(s))
        console_msg = getattr(s, "console_message", agent_msg)
        ActionNotPossible(
            agent_msg,
            console_msg,
            hint=getattr(s, "hint", None),
            context=getattr(s, "context", None),
        )
        current.RESULT.softerror_count += 1
    except HardException as h:
        agent_msg = getattr(h, "agent_message", str(h))
        console_msg = getattr(h, "console_message", agent_msg)
        FormalError(
            agent_msg,
            console_msg,
            hint=getattr(h, "hint", None),
            context=getattr(h, "context", None),
        )
        current.RESULT.harderror_count += 1

def check_id(readable_id: str):
    for entity in World.entities:
        if entity.readable_id == readable_id:
            return entity

    agent_entity = getattr(current.AGENT, "entity", None)
    room = agent_entity.room if agent_entity and agent_entity.room else None
    available_ids: list[str] = []
    if room:
        for ent in room.entities:
            if ent and ent.readable_id:
                available_ids.append(ent.readable_id)

    raise HardException(
        f"No object named '{readable_id}' is available in your current room.",
        console_message=(
            f"Lookup failed for '{readable_id}'. Room "
            f"'{room.name if room else 'unknown'}' currently exposes: "
            f"{', '.join(sorted(available_ids)) or 'no interactive objects'}."
        ),
        hint="Check your latest observation for the correct identifier or move closer to the target.",
        context={
            "requested_id": readable_id,
            "room": room.readable_id if room else None,
            "available_ids": sorted(available_ids),
        },
    )

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
def move_to_object(object_id: str) -> str:
    """The human moves to a position.

    Args:
        object_id (str): the id of the object to move to
    """

    trycatch(lambda: current.AGENT.entity.move_to_object(check_id(object_id)), "moved succesfully")

    return ""
        
@tool
def take_from(what_id: str, from_id: str) -> str:
    """The human picks up an item A from a object B and adds it to their inventory. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the id of item A to be taken.
        from_id (str): the id of object B from which the item is taken or 'FLOOR' for the floor
    """
    
    if(from_id == "FLOOR"):
        trycatch(lambda: current.AGENT.entity.take(check_id(what_id)), f"collected {what_id}")

    else:
        trycatch(lambda: current.AGENT.entity.take_from(check_id(what_id), check_id(from_id)), f"collected {what_id} from {from_id}")

    return ""

@tool
def drop_to(what_id: str, to_id: str) -> str:
    """The human drops an item A from their inventory into or onto another object B. Only use this, when you are nearby B.
    
    Args:
        what_id (str): the the id of the item A to be dropped.
        to_id (str): the id of the object B where the item is placed or 'FLOOR' for the floor
    """

    if(to_id == "FLOOR"):
        trycatch(lambda: current.AGENT.entity.drop(check_id(what_id)), f"dropped {what_id}")
    else:
        trycatch(lambda: current.AGENT.entity.drop_into(check_id(what_id), check_id(to_id)), f"dropped {what_id} into {to_id}")

    return ""

@tool
def interact_with_object(object_id: str, operator: str) -> str:
    """The agent interacts with an object using a specific operator.
    
    Args:
        object_id (str): the the id of the item A to be dropped.
        operator (str): The action to perform. Allowed values: OPEN, CLOSE, GO_THROUGH.
    """

    def helper():
        if(operator == "OPEN"):
            action = ActionTry(ActionType.OPEN)
        elif (operator == "CLOSE"):
            action = ActionTry(ActionType.CLOSE)
        elif (operator == "GO_THROUGH"):
            action = ActionTry(ActionType.USE)
        else:
            raise HardException("unknown operator for this action: {operator}")
        
        return check_id(object_id).on_interact(current.AGENT.entity, action)
    
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

    def helper():
        if(operator == "LOCK"):
            action = ActionTry(ActionType.LOCK, check_id(using_id))
        elif (operator == "UNLOCK"):
            action = ActionTry(ActionType.UNLOCK, check_id(using_id))
        else:
            raise HardException("unknown operator for this action: {operator}")
        
        return check_id(object_id).on_interact(current.AGENT.entity, action)
    
    trycatch(helper, f"succeded with {operator} {object_id}")

    return ""

def run_level(cache, model, level: Level, optimal_steps_multilier: float):

    spec: LevelSpec = level.build()

    console.pretty(console.banner(level.name, char="+", color=console.Color.BLUE))

    agents = []

    for eg in spec.agent_entities:
        entity, prompt = eg

        agent_mem = Memory()
        agent_mem.add_message(Role.SYSTEM, prompt)
        console.pretty(console.bullet(entity.name + "\t[PROMPT:] " + prompt, color=console.Color.BLUE))
        agent = Agent.build(cache.get(model), entity=entity, memory=agent_mem)

        agents.append(agent)


    for i in range(int(level.optimal_steps * optimal_steps_multilier)):
        for agent in agents:
            print(f"Observation: {i+1}")
            room = agent.entity.room
            observation = observe(room, agent.entity)
            agent.entity_step([
                move_to_position, move_to_object,
                take_from, drop_to,
                interact_with_object, interact_with_object_using_item,
                ], observation)
            current.RESULT.observation_count += 1

        if(spec.is_success()):
            print("Finished")
            current.RESULT.success = 1
            return
        
    current.RESULT.success = 0    
    