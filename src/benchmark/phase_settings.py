from __future__ import annotations

from dataclasses import dataclass
from socket import gethostname
from typing import Dict, Sequence

from config import Configuration
from enviroment.levels.level import Level, Levels
from llm.model import Model
from benchmark.model_team import ModelTeam, ModelTeams
from benchmark.run_registry import CONFIGURATIONS


@dataclass
class PhaseDefinition:
    """Describes a matrix of runs (configs × levels × model teams) to enumerate into a TODO list."""

    phase: str
    configs: Sequence[Configuration]
    levels: Sequence[Level | Levels]
    model_teams: Sequence[ModelTeam | ModelTeams]
    reruns: int = 1


@dataclass
class RunnerConfig:
    """Per-device settings for consuming TODO entries (phase allowlist + model-team allowlist)."""

    allowed_model_teams: Sequence[ModelTeam | ModelTeams] | None = None
    allowed_phases: Sequence[str] | None = None


@dataclass
class ResolvedRunnerConfig:
    """Resolved configuration including the concrete phase to run."""

    phase: str
    allowed_model_teams: Sequence[ModelTeam | ModelTeams] | None = None


# Curated phase definitions. Add or adjust entries here to generate multiple TODO lists.
PHASES: Sequence[PhaseDefinition] = (
    PhaseDefinition(
        phase="phase1",
        configs=[
            CONFIGURATIONS["baseline"]
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY
            ],
        model_teams=[ModelTeams.HYBRID_MISTRAL_SMALL],
        reruns=5,
    ),
    PhaseDefinition(
        phase="phase2",
        configs=[CONFIGURATIONS["baseline"]],
        levels=[Levels.DETAILED_INSTRUCT.CUCUMBER_EASY],
        model_teams=[ModelTeams.LOCAL_QWEN_CODER30B],
        reruns=1,
    ),
    PhaseDefinition(
        phase="phase3",
        configs=[CONFIGURATIONS["baseline"]],
        levels=[Levels.DETAILED_INSTRUCT.ONION_EASY],
        model_teams=[ModelTeams.LOCAL_QWEN_CODER30B],
        reruns=1,
    ),
    PhaseDefinition(
        phase="phase4",
        configs=[CONFIGURATIONS["baseline"]],
        levels=[Levels.DETAILED_INSTRUCT.POTATO_EASY],
        model_teams=[ModelTeams.HYBRID_MISTRAL_SMALL],
        reruns=1,
    ),
)

# Map of hostname → runner configuration. Edit the keys to match your devices.
RUNNER_CONFIGS: Dict[str, RunnerConfig] = {
    # Laptop profile: keep both imaginator and realisator local/small.
    "r2d2xxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.HYBRID_MISTRAL_SMALL,
        ],
        allowed_phases=["phase1", "phase2", "phase3"],
    ),
    # PC profile: allow remote imaginator with local realisator.
    "c3poxxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.HYBRID_MISTRAL_SMALL,
        ],
        allowed_phases=["phase1", "phase4"],
    ),
}

# Convenience defaults so other modules can import a single definition.
PHASE_DEFINITION = PHASES[0]
DEFAULT_RUNNER_CONFIG = RunnerConfig(
    allowed_model_teams=PHASE_DEFINITION.model_teams, allowed_phases=[PHASE_DEFINITION.phase]
)


def resolve_runner_config(hostname: str | None = None, phase: str | None = None) -> ResolvedRunnerConfig:
    """Return the runner config for the given hostname (optional phase override)."""

    host = hostname or gethostname()
    runner_config = RUNNER_CONFIGS.get(host, DEFAULT_RUNNER_CONFIG)

    allowed_models = runner_config.allowed_model_teams or DEFAULT_RUNNER_CONFIG.allowed_model_teams
    allowed_phases = runner_config.allowed_phases or DEFAULT_RUNNER_CONFIG.allowed_phases

    if phase:
        if allowed_phases and phase not in allowed_phases:
            known = ", ".join(sorted(allowed_phases))
            raise KeyError(f"Phase '{phase}' not configured for host '{host}'. Known phases: {known}")
        selected_phase = phase
    else:
        if not allowed_phases:
            raise KeyError(f"No phases configured for host '{host}' and no default available.")
        selected_phase = next(iter(allowed_phases))

    return ResolvedRunnerConfig(phase=selected_phase, allowed_model_teams=allowed_models)
