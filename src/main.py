import argparse
from typing import Iterable

from benchmark.benchresult import PerformanceResult
from benchmark.dispatcher import Dispatcher, Run
from config import AgentConfiguration, Configuration, ImaginatorType, ObserveType, PerceptionType, PlanType, PositionType, ReflectType
from enviroment.levels.level import Levels
from llm.model import Model


def _enum_choice(enum_cls) -> Iterable[str]:
    return [member.name.lower() for member in enum_cls]


def _parse_enum(value: str, enum_cls):
    key = value.strip().upper()
    try:
        return enum_cls[key]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Invalid choice '{value}'. Options: {', '.join(_enum_choice(enum_cls))}") from exc


def _parse_level(path: str):
    current = Levels
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError as exc:
            raise argparse.ArgumentTypeError(
                f"Unknown level path '{path}'. Use dot notation like 'VAGUE_INSTRUCT.POTATO_HARD'."
            ) from exc

    from enum import Enum  # local import to avoid polluting module scope

    if isinstance(current, type) and issubclass(current, Enum):
        raise argparse.ArgumentTypeError(f"Level path '{path}' did not resolve to a concrete level")

    return current.value if hasattr(current, "value") else current


def _parse_model(path: str):
    current = Model
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError as exc:
            raise argparse.ArgumentTypeError(
                f"Unknown model path '{path}'. Example: Local.LlamaCpp.Qwen.VANILLA_8B"
            ) from exc

    return current


def main():
    dispatcher = Dispatcher()

    agent_config = AgentConfiguration(
        imaginator = ImaginatorType.QUESTION,
        observe    = ObserveType.ON,
        reflect    = ReflectType.ON,
        plan       = PlanType.DECOMPOSE
    )

    config = Configuration(PerceptionType.ALL, PositionType.RELATIVE, agent=agent_config, temperature=0.0, name="test", seed=-1)

    print(PerformanceResult(Run(
        config,
        #main_model               = Model.Local.Ollama.GPT_OSS_20B,
        main_model               = Model.Local.LlamaCpp.Qwen.VANILLA_8B,
        #imaginator               = Model.Remote.MISTRAL_SMALL_24B,
        level                    = Levels.DETAILED_INSTRUCT.ONION_EASY,
        reruns                   = 30,
        optimal_steps_multiplier = 4.0,
        #extra_model              = Model.Local.Ollama.DOLPHIN3_8B,
        ), 0, 1, 2, 3, 4, 5, 123).toString())

    result1: PerformanceResult = dispatcher.run_single(Run(
        config,
        #main_model               = Model.Local.Ollama.GPT_OSS_20B,
        main_model               = Model.Local.Ollama.Cogito.COGITO_8B,
        #imaginator               = Model.Remote.MISTRAL_SMALL_24B,
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
