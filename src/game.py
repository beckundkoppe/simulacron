import json
from advanced.agent import Agent
from benchmark.level import buildLevel_Potato
from character import current
from character.action import TOOLS, drop_to, move_to_position, take_from, use_door
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
                    msg = "[ACTION] " + result.what
                    color = console.Color.YELLOW.value

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

def run(cache, model):
    agent_mem = Memory()
    agent_mem.add_message(Role.SYSTEM,       
    """
    Your Goal: Find a potato, take it and place it on the table. USE YOUR TOOLCALLS. GIVE TOOLCALLS as toolcalls! NO CONVERSATION
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
        agent.entity_step(TOOLS, observation)

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