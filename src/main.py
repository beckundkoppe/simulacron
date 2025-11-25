import argparse
from typing import Iterable

from benchmark.benchresult import RunResult
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
    parser = argparse.ArgumentParser(description="Run Simulacron benchmarks from the CLI.")
    parser.add_argument("--level", default="VAGUE_INSTRUCT.POTATO_HARD", help="Level path (e.g. VAGUE_INSTRUCT.POTATO_HARD)")
    parser.add_argument("--reruns", type=int, default=1, help="How many times to repeat the run")
    parser.add_argument("--optimal-steps-multiplier", type=float, default=4.0, help="Multiplier for the optimal step count")
    parser.add_argument("--main-model", default="Local.LlamaCpp.Qwen.VANILLA_8B", help="Model path for the agent")
    parser.add_argument("--imaginator-model", default="Remote.MISTRAL_SMALL_24B", help="Model path for the imaginator")
    parser.add_argument("--extra-model", default=None, help="Optional additional model path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--name", default="run", help="Name for this configuration")

    parser.add_argument("--imaginator", default="question", type=lambda v: _parse_enum(v, ImaginatorType), choices=_enum_choice(ImaginatorType))
    parser.add_argument("--observe", default="on", type=lambda v: _parse_enum(v, ObserveType), choices=_enum_choice(ObserveType))
    parser.add_argument("--reflect", default="on", type=lambda v: _parse_enum(v, ReflectType), choices=_enum_choice(ReflectType))
    parser.add_argument("--plan", default="structured", type=lambda v: _parse_enum(v, PlanType), choices=_enum_choice(PlanType))
    parser.add_argument("--perception", default="all", type=lambda v: _parse_enum(v, PerceptionType), choices=_enum_choice(PerceptionType))
    parser.add_argument("--position", default="relative", type=lambda v: _parse_enum(v, PositionType), choices=_enum_choice(PositionType))

    args = parser.parse_args()

    dispatcher = Dispatcher()

    agent_config = AgentConfiguration(
        imaginator=args.imaginator,
        observe=args.observe,
        reflect=args.reflect,
        plan=args.plan,
    )

    config = Configuration(
        args.perception,
        args.position,
        agent=agent_config,
        temperature=args.temperature,
        name=args.name,
        seed=args.seed,
    )

    level = _parse_level(args.level)
    main_model = _parse_model(args.main_model)
    imaginator_model = _parse_model(args.imaginator_model) if args.imaginator is not ImaginatorType.OFF else None
    extra_model = _parse_model(args.extra_model) if args.extra_model else None

    result: RunResult = dispatcher.run_single(
        Run(
            config,
            main_model=main_model,
            imaginator=imaginator_model,
            level=level,
            reruns=args.reruns,
            optimal_steps_multiplier=args.optimal_steps_multiplier,
            extra_model=extra_model,
        )
    )

    print("\n\n\nReruns:", args.reruns)
    print(result.toString())


if __name__ == "__main__":
    main()
