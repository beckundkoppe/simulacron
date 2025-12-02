from benchmark.benchresult import PerformanceResult
from benchmark.dispatcher import Dispatcher, Run
from benchmark.model_team import ModelTeam
from config import AgentConfiguration, Configuration, ActionType, ObserveType, PerceptionType, PlanType, PositionType, ReflectType, TrialType
from enviroment.levels.level import Levels
from llm.model import Model


def main():
    dispatcher = Dispatcher()

    agent_config = AgentConfiguration(
        plan       = PlanType.DECOMPOSE,
        observe    = ObserveType.OFF,
        trial      = TrialType.OFF,
        action     = ActionType.IMG_RETRY,
        reflect    = ReflectType.OFF,
    )

    config = Configuration(agent_config, PerceptionType.ALL, PositionType.RELATIVE, temperature=0.0, name="test", seed=-1)

    model_team = ModelTeam(
        #realisator=Model.Remote.MISTRAL_SMALL_24B,
        #realisator=Model.Local.Ollama.Qwen3.VANILLA_8B,
        realisator=Model.Remote.GPT_OSS_120B,
        #extra=Model.Local.Ollama.DOLPHIN3_8B,
    )

    result1: PerformanceResult = dispatcher.run_single(
        Run(
            config,
            model_team=model_team,
            level=Levels.DETAILED_INSTRUCT.ONION_EASY,
            reruns=30,
            optimal_steps_multiplier=4.0,
        )
    )

    print("\n\n\nReruns: 20")
    print(result1.toString())


if __name__ == "__main__":
    main()
