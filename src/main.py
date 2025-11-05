# test_main.py

from advanced.agent import Agent
from benchmark import benchmark
from benchmark.level import buildLevel_Potato
from character import action
from character.action import move_to_position, use_door
import debug.console as console
from enviroment.entity import Entity
from enviroment.entity_controller import EntityController
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

def observe(room: Room, observer: Entity) -> str:
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
    data["your_observation"] = room.perceive(observer.uuid, Depth.OMNISCIENT)
    return str(data)

def agent_step(agent: Agent, tron_entity: Entity):
    room = World.get_room(tron_entity.room)
    agent.register_tools([move_to_position, use_door])
    agent.invoke(observe(room, tron_entity))
    
def main():
    cache = Cache()
    cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B)

    agent_mem = Memory()
    agent_mem.add_message(Role.SYSTEM,       
    """
    Use tools only
    """)

    agent = Agent.build(cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B), memory=agent_mem)
    
    perception = ObserverPerception()
    tron = Entity("tron", Position(0.0, 0.0), perception=perception)
    level = buildLevel_Potato(tron)

    action.CONTROL = EntityController(tron)

    while(True):
        agent_step(agent, tron)
        #agent.update()
        #level.update()

if __name__ == "__main__":
    main()
