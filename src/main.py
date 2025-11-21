# test_main.py

from benchmark.benchresult import RunResult
from benchmark.dispatcher import Dispatcher, Run
from config import AgentConfiguration, Configuration, ImaginatorType, ObserveType, PerceptionType, PlanType, PositionType, ReflectType
from enviroment.levels.level import Levels
import game
from llm.cache import Cache
from llm.model import Model
    
def main():
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

    dispatcher = Dispatcher()

    agent_config = AgentConfiguration(
        imaginator = ImaginatorType.QUESTION,
        observe    = ObserveType.ON,
        reflect    = ReflectType.ON,
        plan       = PlanType.PLAN
    )

    config = Configuration(PerceptionType.ALL, PositionType.RELATIVE, agent=agent_config, temperature=0.0, name="test", seed=-1)

    result1: RunResult = dispatcher.run_single(Run(
        config,
        main_model               = Model.Local.Ollama.GPT_OSS_20B,
        #imaginator               = Model.Remote.MISTRAL_SMALL_24B,
        level                    = Levels.VAGUE_INSTRUCT.POTATO_HARD.value,
        reruns                   = 30,
        optimal_steps_multiplier = 4.0,
        #extra_model              = Model.Local.Ollama.DOLPHIN3_8B,
        )
    )

    print("\n\n\nReruns: 20")
    print(result1.toString())

    #Only use toolcalls. For toolcalls, use the following syntax: {"name": "<tool_name>", "args": {"<arg_key>": "<arg_value>"}}



if __name__ == "__main__":
    main()
