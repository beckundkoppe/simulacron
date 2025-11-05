# test_main.py

import json
from advanced.agent import Agent
from benchmark import benchmark
from benchmark.level import buildLevel_Potato
from character import action
from character.action import TOOLS, move_to_position, use_door
import debug.console as console
from enviroment.entity import AgentEntity, ContainerEntity, Entity
from enviroment.interaction import Depth, ObserverPerception, PerceptionEnviroment
from enviroment.room import Position, Room
from enviroment.world import World
from llm.cache import Cache
from llm.memory.memory import Memory
from llm.model import Model
from llm.runner import Role

def observe_all():
    for ent in World.entities:
        entity = World.get_entity(ent)
        if entity.perception is None:
            continue
        
        if entity.room is None:
            continue
        
        room = World.get_room(entity.room)

        observations = observe(room, entity)
        console.json_dump(observations)

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

def agent_step(agent: Agent, tron_entity: Entity):
    room = World.get_room(tron_entity.room)
    agent.register_tools(TOOLS)
    agent.invoke(observe(room, tron_entity))
    
def main():
    cache = Cache()
    #cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B)
    #cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    #cache.get(Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B)
    #cache.get(Model.Local.Ollama.HYBRID_GPT_OSS_20B)


    agent_mem = Memory()
    agent_mem.add_message(Role.SYSTEM,       
    """
    Your Goal: Find a potato, take it and place it on the table. TOOLCALLS ONLY! NO TEXT!
    """
    #"""
    #Use tools only.
    #Use this syntax for tools:
    #{'name': '<functionName>', 'args': {'<key>': '<value>'}
    #"""
    )

    agent_mem.debug_print()

    agent = Agent.build(cache.get(Model.Local.Ollama.HYBRID_GPT_OSS_20B), memory=agent_mem)
    
    perception = ObserverPerception()
    tron = AgentEntity("tron", Position(0.0, 0.0), perception=perception)
    level = buildLevel_Potato(tron)

    action.CONTROL = tron

    while(True):
        agent_step(agent, tron)

        if(check_win()):
            print("Finished")
            break
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

if __name__ == "__main__":
    main()
