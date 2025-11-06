import json
from advanced.agent import Agent
from advanced.tool import tool
from benchmark.level import buildLevel_Potato
from character import current
from character.exception import HardException, SoftException
from character.resultbuffer import ActionNotPossible, FormalError, Resultbuffer, Success
from debug import console
from enviroment.entity import AgentEntity, ContainerEntity
from enviroment.interaction import Depth, ObserverPerception
from enviroment.position import Position
from enviroment.room import Room
from enviroment.world import World
from llm.memory.memory import Memory, Role

#def observe_all():
#    for ent in World.entities:
#        entity = World.get_entity(ent)
#        if entity.perception is None:
#            continue
#        
#        if entity.room is None:
#            continue
#        
#        room = World.get_room(entity.room)
#
#        observations = observe(room, entity)
#        console.json_dump(observations)
#

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
    data["your_observation"] = room.perceive(observer.uuid, Depth.OMNISCIENT)
    return json.dumps(data)

def test_run():
    class Agent:
        def __init__(self, entity):
            self.entity = entity
        
        def process_results(self):
            for result in Resultbuffer.buffer:
                if isinstance(result, FormalError):
                    msg = "[FORMAL ERROR] " + result.what
                    color = console.Color.RED.value
                if isinstance(result, ActionNotPossible):
                    msg = "[ACTION FAILURE] " + result.what
                    color = console.Color.RED.value
                if isinstance(result, Success):
                    return
                    #msg = "[ACTION] " + result.what
                    #color = console.Color.YELLOW.value

                console.pretty(
                    console.bullet(f"[toolcall]\t{msg}", color),
                )
            Resultbuffer.buffer.clear()
            

    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    buildLevel_Potato(tron)

    agent = Agent(tron)
    current.AGENT = agent

    room = World.get_room(tron.room)

    observation = observe(World.get_room(tron.room), tron)
    print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observation))!s}", color=console.Color.CYAN))

    move_to_position("2.0","2.0")
    agent.process_results()

    use_door("door_6")
    agent.process_results()

    observation = observe(World.get_room(tron.room), tron)
    print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observation))!s}", color=console.Color.CYAN))


    take_from("potato_8", "chest_10")
    agent.process_results()

    observation = observe(World.get_room(tron.room), tron)
    print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observation))!s}", color=console.Color.CYAN))

    use_door("door_7")
    agent.process_results()

    observation = observe(World.get_room(tron.room), tron)
    print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observation))!s}", color=console.Color.CYAN))

    drop_to("potato_8", "table_5")
    agent.process_results()
    move_to_position("1.0", "1.0")
    agent.process_results()
    observation = observe(World.get_room(tron.room), tron)
    print(console.bullet_multi(f"[user] {console.dump_limited(json.loads(observation))!s}", color=console.Color.CYAN))

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

def run(cache, model):
    agent_mem = Memory()
    agent_mem.add_message(Role.SYSTEM,       
    """
    Goal: Find a potato, take it and place it on the table. GIVE the next toolcall - nothing more
    """
    )
    agent_mem.debug_print()

    perception = ObserverPerception()
    tron = AgentEntity("tron", perception, pos=Position(0.0, 0.0))
    level = buildLevel_Potato(tron)

    agent = Agent.build(cache.get(model), entity=tron, memory=agent_mem)

    while(True):
        room = World.get_room(agent.entity.room)

        observation = observe(room, agent.entity)
        agent.entity_step([move_to_position, use_door, take_from, drop_to], observation)

        if(check_win()):
            print("Finished")
            break

        #input()
        #agent.update()
        #level.update()

def check_win() -> bool:
    potato = None
    table: ContainerEntity = None

    for uuid in World.entities:
        ent = World.get_entity(uuid)
        if(ent.name == "potato"):
            potato = ent
        if(ent.name == "table"):
            table = ent
    
    assert potato != None, "No potato"
    assert table != None, "No table"
    
    return potato.uuid in table.children