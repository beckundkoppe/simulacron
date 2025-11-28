from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmark.run import Run
from config import (
    ActionType,
    AgentConfiguration,
    Configuration,
    ObserveType,
    PerceptionType,
    PlanType,
    PositionType,
    ReflectType,
    TrialType,
)
from enviroment.levels.level import Level, Levels
from llm.model import Model, ModelSpec


# Preset configurations that can be referenced by name in the phase files.
DEFAULT_AGENT = AgentConfiguration(
    plan=PlanType.DECOMPOSE,
    trial=TrialType.ON,
    action=ActionType.IMG_RETRY,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
)

CONFIGURATIONS: Dict[str, Configuration] = {
    "baseline": Configuration(
        agent=DEFAULT_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="baseline",
        seed=-1,
    ),
}


def _walk_enum(enum_cls: type[Enum]) -> Iterable[Enum]:
    for member in enum_cls:
        yield member
        value = member.value
        if isinstance(value, type) and issubclass(value, Enum):
            yield from _walk_enum(value)


def collect_levels() -> Dict[str, Enum]:
    level_map: Dict[str, Enum] = {}
    for member in _walk_enum(Levels):
        value = member.value
        if isinstance(value, Level):
            level_map[value.getName()] = member
    return level_map


def collect_models() -> Dict[str, Enum]:
    model_map: Dict[str, Enum] = {}
    for member in _walk_enum(Model):
        value = member.value
        if isinstance(value, ModelSpec):
            model_map[value.name] = member
    return model_map


def collect_configurations() -> Dict[str, Configuration]:
    return CONFIGURATIONS


def normalize_configurations(
    configs: Sequence[str | Configuration] | None, fallback: Sequence[Configuration]
) -> List[Configuration]:
    lookup = collect_configurations()
    if configs is None:
        return list(fallback)

    normalized: List[Configuration] = []
    for config in configs:
        if isinstance(config, Configuration):
            normalized.append(config)
            continue

        name = str(config)
        if name not in lookup:
            raise KeyError(f"Unknown configuration '{name}'. Known: {', '.join(sorted(lookup))}")
        normalized.append(lookup[name])

    return normalized


def _level_enum_from_input(level_lookup: Dict[str, Enum], level: str | Enum | Level) -> Enum:
    if isinstance(level, Enum):
        if isinstance(level.value, Level):
            return level
        # Nested enums (e.g., Levels.DETAILED_INSTRUCT) contain Level instances downstream
        value = level.value
        if isinstance(value, Enum) and isinstance(value.value, Level):
            return value

    if isinstance(level, Level):
        level_name = level.getName()
    else:
        level_name = str(level)

    if level_name not in level_lookup:
        raise KeyError(f"Unknown level '{level_name}'. Known: {', '.join(sorted(level_lookup))}")
    return level_lookup[level_name]


def normalize_levels(
    levels: Sequence[str | Enum | Level] | None, fallback: Sequence[Enum]
) -> List[Enum]:
    level_lookup = collect_levels()
    if levels is None:
        return list(fallback)

    return [_level_enum_from_input(level_lookup, lvl) for lvl in levels]


def _model_name(model: str | Enum | ModelSpec) -> str:
    if isinstance(model, Enum):
        value = model.value
        if isinstance(value, ModelSpec):
            return value.name
        if isinstance(value, Enum) and isinstance(value.value, ModelSpec):
            return value.value.name
    if isinstance(model, ModelSpec):
        return model.name
    return str(model)


def normalize_models(models: Sequence[str | Enum | ModelSpec] | None, fallback: Sequence[Enum]) -> List[Enum]:
    model_lookup = collect_models()
    if models is None:
        return list(fallback)

    normalized: List[Enum] = []
    for model in models:
        name = _model_name(model)
        if name not in model_lookup:
            raise KeyError(f"Unknown model '{name}'. Known: {', '.join(sorted(model_lookup))}")
        normalized.append(model_lookup[name])
    return normalized


def parse_filename(filename: str) -> Tuple[str, str, str, int]:
    base = Path(filename).name
    if base.endswith(".json"):
        base = base[:-5]

    # level may contain underscores, so split from the right
    try:
        base, rerun_str = base.rsplit("_", 1)
        base, config_name = base.rsplit("_", 1)
        level_name, model_name = base.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid run filename: {filename}") from exc

    return level_name, model_name, config_name, int(rerun_str)


def build_run(level_name: str, model_name: str, config_name: str, rerun_index: int) -> Run:
    level_lookup = collect_levels()
    model_lookup = collect_models()
    config_lookup = collect_configurations()

    if level_name not in level_lookup:
        raise KeyError(f"Unknown level '{level_name}' in filename")
    if model_name not in model_lookup:
        raise KeyError(f"Unknown model '{model_name}' in filename")
    if config_name not in config_lookup:
        raise KeyError(f"Unknown configuration '{config_name}' in filename")

    return Run(
        configuration=config_lookup[config_name],
        main_model=model_lookup[model_name],
        level=level_lookup[level_name],
        reruns=rerun_index + 1,
        optimal_steps_multiplier=1.0,
        imaginator=None,
        extra_model=None,
    )


def filter_by_models(entries: List[str], allowed_models: Sequence[str | Enum | ModelSpec] | None) -> List[str]:
    if not allowed_models:
        return entries
    allowed_set = {_model_name(model) for model in allowed_models}
    filtered: List[str] = []
    for entry in entries:
        _, model_name, _, _ = parse_filename(entry)
        if model_name in allowed_set:
            filtered.append(entry)
    return filtered
