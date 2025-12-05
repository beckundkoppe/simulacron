from __future__ import annotations

from dataclasses import replace
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Any

from benchmark.model_team import ModelTeam, ModelTeams
from benchmark.run import Run
from config import (
    ActionType,
    AgentConfiguration,
    Configuration,
    MemoryType,
    ObserveType,
    PerceptionType,
    PlanType,
    PositionType,
    ReflectType,
    TrialType,
)
from enviroment.levels.level import Level, Levels
from llm.model import Model, ModelSpec


# Preset configuration specs that can be referenced by name in the phase files.
NAIVE_AGENT = AgentConfiguration(
    plan=PlanType.OFF,
    trial=TrialType.OFF,
    action=ActionType.DIRECT,
    observe=ObserveType.OFF,
    reflect=ReflectType.OFF,
    memory_type=MemoryType.SIMPLE,
)

IMAGINATOR_AGENT = AgentConfiguration(
    plan=PlanType.OFF,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.OFF,
    reflect=ReflectType.OFF,
    memory_type=MemoryType.SIMPLE,
)

A_AGENT = AgentConfiguration(
    plan=PlanType.OFF,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

B_AGENT = AgentConfiguration(
    plan=PlanType.FREE,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

FREE_AGENT = AgentConfiguration(
    plan=PlanType.FREE,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

STEP_AGENT = AgentConfiguration(
    plan=PlanType.STEP,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

TREE_AGENT = AgentConfiguration(
    plan=PlanType.DECOMPOSE,
    trial=TrialType.OFF,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

TRIAL_AGENT = AgentConfiguration(
    plan=PlanType.DECOMPOSE,
    trial=TrialType.ON,
    action=ActionType.IMAGINATOR,
    observe=ObserveType.ON,
    reflect=ReflectType.ON,
    memory_type=MemoryType.SUPER,
)

CONFIGURATIONS: Dict[str, Configuration] = {
    "baseline-naive": Configuration(
        agent=NAIVE_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="baseline-naive",
    ),
    "baseline-img": Configuration(
        agent=IMAGINATOR_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="baseline-img",
    ),
    "a-agent": Configuration(
        agent=A_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="a-agent",
    ),
    "b-agent": Configuration(
        agent=B_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="b-agent",
    ),
    "free-agent": Configuration(
        agent=FREE_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="free-agent",
    ),
    "step-agent": Configuration(
        agent=STEP_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="step-agent",
    ),
    "trial-agent": Configuration(
        agent=TRIAL_AGENT,
        perception=PerceptionType.ALL,
        position=PositionType.RELATIVE,
        temperature=0.0,
        name="trial-agent",
    ),
}


def _walk_enum(obj: object, _seen: set[type] | None = None) -> Iterable[Enum]:
    """
    Recursively yield enum members and nested enums.

    Some enums in this codebase contain nested enum classes even when they
    already declare members. We always traverse both the members and any nested
    enums to ensure nothing is skipped.
    """
    if _seen is None:
        _seen = set()

    # If we receive an Enum subclass (e.g., Levels or Model.Local.LlamaCpp), walk
    # its members and then any nested Enum classes declared on it.
    if isinstance(obj, type) and issubclass(obj, Enum):
        if obj in _seen:
            return
        _seen.add(obj)
        for member in obj:
            yield from _walk_enum(member, _seen)
        for value in obj.__dict__.values():
            if isinstance(value, type) and issubclass(value, Enum):
                yield from _walk_enum(value, _seen)
        return

    # If we receive an Enum member, inspect its value.
    if isinstance(obj, Enum):
        yield obj
        value = obj.value
        if isinstance(value, type) and issubclass(value, Enum):
            yield from _walk_enum(value, _seen)
        return


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


def _resolve_configuration(config_token: str) -> Configuration:
    """
    Map a configuration token from the filename to a Configuration object.

    Supports both named configurations (e.g., 'baseline-naive') and legacy
    filenames that encode a raw temperature value (e.g., '0.0'). Temperature
    tokens are applied on top of the first known configuration to preserve
    other settings while matching the filename token for output paths.
    """
    lookup = collect_configurations()
    if config_token in lookup:
        return lookup[config_token]

    try:
        temperature = float(config_token)
    except ValueError as exc:
        raise KeyError(f"Unknown configuration '{config_token}' in filename") from exc

    base_config = next(iter(lookup.values()))
    return replace(base_config, temperature=temperature, name=config_token)


def _resolve_model_enum(model_lookup: Dict[str, Enum], model_name: str) -> Enum:
    if model_name not in model_lookup:
        raise KeyError(f"Unknown model '{model_name}'. Known: {', '.join(sorted(model_lookup))}")
    return model_lookup[model_name]

def _collect_model_teams() -> List[ModelTeam]:
    teams: List[ModelTeam] = []
    for member in _walk_enum(ModelTeams):
        if isinstance(member.value, ModelTeam):
            teams.append(member.value)
    return teams


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


def normalize_model_teams(
    model_teams: Sequence[ModelTeam | Enum] | None,
    fallback: Sequence[ModelTeam | Enum],
) -> List[ModelTeam]:
    if model_teams is None:
        model_teams = fallback

    normalized: List[ModelTeam] = []
    for team in model_teams:
        if isinstance(team, ModelTeam):
            normalized.append(team)
        elif isinstance(team, Enum) and isinstance(team.value, ModelTeam):
            normalized.append(team.value)
        else:
            raise TypeError(f"Unsupported model team definition: {team!r}")
    return normalized


def parse_model_token(model_token: str, model_lookup: Dict[str, Enum]) -> ModelTeam:
    if "+" in model_token:
        real_name, imag_name = model_token.split("+", 1)
        real_model = _resolve_model_enum(model_lookup, real_name)
        imaginator_model = _resolve_model_enum(model_lookup, imag_name)
    else:
        real_model = _resolve_model_enum(model_lookup, model_token)
        imaginator_model = real_model

    return ModelTeam(realisator=real_model, imaginator=imaginator_model)


def _parse_model_team_token(team_token: str) -> ModelTeam:
    lookup: Dict[str, ModelTeam] = {}
    for team in _collect_model_teams():
        for key in (team.tag, team.label(), team.token(), team.realisator.value.name):
            if key:
                lookup.setdefault(key, team)

    if team_token in lookup:
        return lookup[team_token]

    # Fallback: allow legacy filenames that encode raw model names
    model_lookup = collect_models()
    return parse_model_token(team_token, model_lookup)


def parse_filename(filename: str) -> Tuple[str, ModelTeam, str, int]:
    base = Path(filename).name
    if base.endswith(".json"):
        base = base[:-5]

    # level may contain underscores, so split from the right
    try:
        base, rerun_str = base.rsplit("_", 1)
        base, config_name = base.rsplit("_", 1)
        level_name, model_token = base.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid run filename: {filename}") from exc

    model_team = _parse_model_team_token(model_token)

    return level_name, model_team, config_name, int(rerun_str)


def build_run(level_name: str, model_team: ModelTeam | Enum, config_name: str, rerun_index: int) -> Run:
    level_lookup = collect_levels()

    if level_name not in level_lookup:
        raise KeyError(f"Unknown level '{level_name}' in filename")

    if isinstance(model_team, ModelTeam):
        resolved_team = model_team
    elif isinstance(model_team, Enum) and isinstance(model_team.value, ModelTeam):
        resolved_team = model_team.value
    else:
        raise TypeError("model_team must be a ModelTeam or ModelTeams enum")

    return Run(
        configuration=_resolve_configuration(config_name),
        model_team=resolved_team,
        level=level_lookup[level_name],
        reruns=rerun_index + 1,
        optimal_steps_multiplier=3.0,
    )


def filter_by_models(entries: List[str], allowed_models: Sequence[ModelTeam | Enum] | None) -> List[str]:
    if not allowed_models:
        return entries

    normalized = normalize_model_teams(allowed_models, [])
    # Preserve priority order from the allowlist
    label_order = [team.label() for team in normalized]
    label_set = set(label_order)

    grouped: Dict[str, List[str]] = {label: [] for label in label_order}
    for entry in entries:
        _, model_team, _, _ = parse_filename(entry)
        label = model_team.label()
        if label in label_set:
            grouped[label].append(entry)

    filtered: List[str] = []
    for label in label_order:
        filtered.extend(grouped.get(label, []))

    return filtered
