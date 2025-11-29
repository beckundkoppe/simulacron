from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from benchmark.model_team import ModelTeam
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


def _resolve_model_enum(model_lookup: Dict[str, Enum], model: str | Enum | ModelSpec) -> Enum:
    name = _model_name(model)
    if name not in model_lookup:
        raise KeyError(f"Unknown model '{name}'. Known: {', '.join(sorted(model_lookup))}")
    return model_lookup[name]


def _coerce_model_team(model_lookup: Dict[str, Enum], team: object) -> ModelTeam:
    if isinstance(team, ModelTeam):
        return team

    if isinstance(team, Enum) and isinstance(team.value, ModelTeam):
        return team.value

    if isinstance(team, tuple) and len(team) in (2, 3):
        real_input, imag_input, *rest = team
        extra_input = rest[0] if rest else None
        real_model = _resolve_model_enum(model_lookup, real_input)
        imaginator_model = _resolve_model_enum(model_lookup, imag_input) if imag_input is not None else real_model
        extra_model = _resolve_model_enum(model_lookup, extra_input) if extra_input is not None else None
        return ModelTeam(realisator=real_model, imaginator=imaginator_model, extra=extra_model)

    if isinstance(team, (str, Enum, ModelSpec)):
        model_enum = _resolve_model_enum(model_lookup, team)
        return ModelTeam(realisator=model_enum, imaginator=model_enum)

    raise TypeError(f"Unsupported model team definition: {team!r}")


def normalize_model_teams(
    model_teams: Sequence[
        str | Enum | ModelSpec | ModelTeam | tuple[object, object] | tuple[object, object, object]
    ]
    | None,
    fallback: Sequence[ModelTeam | Enum | ModelSpec],
) -> List[ModelTeam]:
    model_lookup = collect_models()
    if model_teams is None:
        return [_coerce_model_team(model_lookup, team) for team in fallback]

    return [_coerce_model_team(model_lookup, team) for team in model_teams]


def normalize_models(models: Sequence[str | Enum | ModelSpec] | None, fallback: Sequence[Enum]) -> List[Enum]:
    """Backward-compatible wrapper to normalize single-model lists."""

    teams = normalize_model_teams(models, [ModelTeam(realisator=model) for model in fallback])
    return [team.realisator for team in teams]


def parse_model_token(model_token: str, model_lookup: Dict[str, Enum] | None = None) -> ModelTeam:
    lookup = model_lookup or collect_models()
    if "+" in model_token:
        real_name, imag_name = model_token.split("+", 1)
        real_model = _resolve_model_enum(lookup, real_name)
        imaginator_model = _resolve_model_enum(lookup, imag_name)
    else:
        real_model = _resolve_model_enum(lookup, model_token)
        imaginator_model = real_model

    return ModelTeam(realisator=real_model, imaginator=imaginator_model)


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

    model_team = parse_model_token(model_token)

    return level_name, model_team, config_name, int(rerun_str)


def build_run(level_name: str, model_team: str | ModelTeam, config_name: str, rerun_index: int) -> Run:
    level_lookup = collect_levels()
    config_lookup = collect_configurations()
    model_lookup = collect_models()

    if level_name not in level_lookup:
        raise KeyError(f"Unknown level '{level_name}' in filename")
    if config_name not in config_lookup:
        raise KeyError(f"Unknown configuration '{config_name}' in filename")

    if isinstance(model_team, ModelTeam):
        resolved_team = ModelTeam(
            realisator=_resolve_model_enum(model_lookup, model_team.realisator),
            imaginator=_resolve_model_enum(model_lookup, model_team.imaginator or model_team.realisator),
            extra=_resolve_model_enum(model_lookup, model_team.extra) if model_team.extra is not None else None,
        )
    else:
        resolved_team = parse_model_token(str(model_team), model_lookup)

    return Run(
        configuration=config_lookup[config_name],
        model_team=resolved_team,
        level=level_lookup[level_name],
        reruns=rerun_index + 1,
        optimal_steps_multiplier=1.0,
    )


def filter_by_models(entries: List[str], allowed_models: Sequence[str | Enum | ModelSpec | ModelTeam] | None) -> List[str]:
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
