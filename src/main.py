from benchmark.benchresult import PerformanceResult
from benchmark.dispatcher import Dispatcher, Run
from config import AgentConfiguration, Configuration, ActionType, ObserveType, PerceptionType, PlanType, PositionType, ReflectType, TrialType
from enviroment.levels.level import Levels
from llm.model import Model


def main():
    dispatcher = Dispatcher()

    agent_config = AgentConfiguration(
        plan       = PlanType.DECOMPOSE,
        observe    = ObserveType.ON,
        trial      = TrialType.ON,
        action     = ActionType.IMG_RETRY,
        reflect    = ReflectType.ON,
    )

    config = Configuration(agent_config, PerceptionType.ALL, PositionType.RELATIVE, temperature=0.0, name="test", seed=-1)

    result1: PerformanceResult = dispatcher.run_single(Run(
        config,
        #main_model               = Model.Remote.MISTRAL_SMALL_24B,
        #main_model               = Model.Local.Ollama.Qwen3.VANILLA_8B,
        main_model               = Model.Remote.QWEN3,
        level                    = Levels.DETAILED_INSTRUCT.ONION_EASY,
        reruns                   = 30,
        optimal_steps_multiplier = 4.0,
        #extra_model              = Model.Local.Ollama.DOLPHIN3_8B,
        )
    )

    print("\n\n\nReruns: 20")
    print(result1.toString())


if __name__ == "__main__":
    main()
