# test_main.py

from benchmark.benchresult import RunResult
from benchmark.dispatcher import Dispatcher, Run
from config import AgentConfiguration, Configuration, ObserveType, PerceptionType, PlanType, PositionType, ReflectType
from enviroment.levels.level import Levels
import game
from llm.cache import Cache
from llm.model import Model
    
def main():
    cache = Cache()
    #cache.get(Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B)
    #cache.get(Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B)
    #cache.get(Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B)
    #cache.get(Model.Local.Ollama.HYBRID_GPT_OSS_20B)

    #cache.get(Model.Local.Ollama.Qwen3.CODER_30B)
    #cache.get(Model.Local.Ollama.Qwen3.VANILLA_30B)
    #cache.get(Model.Local.Ollama.Qwen3.VANILLA_14B)
    #cache.get(Model.Local.Ollama.Qwen3.VANILLA_8B)
    #cache.get(Model.Local.Ollama.Qwen3.VANILLA_4B)
    #cache.get(Model.Local.Ollama.Qwen3.INSTRUCT_30B)

    #TEST#############################
    #game.test_run()
    #TEST#############################

    dispatcher = Dispatcher(cache)

    agent_config = AgentConfiguration(
        imaginator = False,
        observe    = ObserveType.ON,
        reflect    = ReflectType.ON,
        plan       = PlanType.OFF
    )

    config = Configuration(PerceptionType.ALL, PositionType.CHESSBOARD, agent=agent_config, temperature=0.0, name="test")

    result1: RunResult = dispatcher.run_single(Run(
        config,
        Model.Local.Ollama.Qwen3.VANILLA_8B,
        Levels.DETAILED_INSTRUCT.POTATO_EASY.value,
        10,
        3.0,
        extra_model=Model.Local.Ollama.Granite.GRANITE4_3B,
        )
    )

    print("\n\n\nReruns: 4")
    print(result1.toString())

    #Only use toolcalls. For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}



if __name__ == "__main__":
    main()
