from __future__ import annotations

from dataclasses import dataclass
from socket import gethostname
from typing import Dict, Sequence

from config import Configuration
from enviroment.levels.level import Level, Levels
from llm.model import Model
from benchmark.run_registry import CONFIGURATIONS


@dataclass
class PhaseDefinition:
    """Describes a matrix of runs to enumerate into a TODO list."""

    phase: str
    configs: Sequence[Configuration]
    levels: Sequence[Level | Levels]
    models: Sequence[Model]
    reruns: int = 1


@dataclass
class RunnerConfig:
    """Per-device settings for consuming TODO entries."""

    phase: str
    allowed_models: Sequence[Model] | None = None


# Curated phase definitions. Add or adjust entries here to generate multiple TODO lists.
PHASES: Sequence[PhaseDefinition] = (
    PhaseDefinition(
        phase="phase1",
        configs=[CONFIGURATIONS["baseline"]],
        levels=[Levels.DETAILED_INSTRUCT.value.CARROT_EASY],
        models=[Model.Local.value.LlamaCpp.value.Deepseek.value.CODER_V2_16B_Q8],
        reruns=1,
    ),
)

# Map of hostname → runner configuration. Edit the keys to match your devices.
RUNNER_CONFIGS: Dict[str, RunnerConfig] = {
    gethostname(): RunnerConfig(
        phase=PHASES[0].phase,
        allowed_models=PHASES[0].models,
    ),
}

# Convenience defaults so other modules can import a single definition.
PHASE_DEFINITION = PHASES[0]
RUNNER_CONFIG = RUNNER_CONFIGS.get(
    gethostname(), RunnerConfig(phase=PHASE_DEFINITION.phase, allowed_models=PHASE_DEFINITION.models)
)


def resolve_runner_config(hostname: str | None = None) -> RunnerConfig:
    """Return the runner config for the given hostname (defaults to current machine)."""

    host = hostname or gethostname()
    return RUNNER_CONFIGS.get(host, RUNNER_CONFIG)
