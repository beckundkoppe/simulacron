# test_main.py



from benchmark.benchresult import RunResult
from benchmark.dispatcher import Dispatcher, Run
from config import Configuration, PerceptionType, PlanType, PositionType
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
    config = Configuration(PerceptionType.FULL, PositionType.RELATIVE, True, True, PlanType.REPLAN, 0.0, "test")
    result: RunResult = dispatcher.run_single(Run(
        config,
        Model.Local.Ollama.Qwen3.VANILLA_14B,
        Levels.POTATO_HARD.value,
        1,
        2.0
        )
    )
    print(result.toString())



if __name__ == "__main__":
    main()
