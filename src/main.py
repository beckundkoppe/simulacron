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
import game
from llm.cache import Cache
from llm.memory.memory import Memory
from llm.model import Model
from llm.runner import Role
    
def main():
    cache = Cache()
    #cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B)
    #cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    #cache.get(Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B)
    #cache.get(Model.Local.Ollama.HYBRID_GPT_OSS_20B)

    #TEST#############################
    #game.test_run()
    #TEST#############################

    game.run(cache, Model.Local.Ollama.HYBRID_GPT_OSS_20B)

if __name__ == "__main__":
    main()
