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
        phase="backend",
        configs=[
            CONFIGURATIONS["baseline-img"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            ],
        model_teams=[
            ModelTeams.Local.QWEN_8B_LLAMA,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_LLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,
            ModelTeams.Local.QWEN_4B_Q4_LLAMA,
            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,
        ],
        reruns=5,
    ),

    PhaseDefinition(
        phase="baseline-solo",
        configs=[
            CONFIGURATIONS["baseline-naive"],
            CONFIGURATIONS["baseline-img"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            #Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            #Levels.DETAILED_INSTRUCT.TOMATO_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            #Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            #Levels.VAGUE_INSTRUCT.TOMATO_HARD,

            ],
        model_teams=[
            ModelTeams.Remote.MAGISTRAL_SMALL,
            ModelTeams.Remote.GPT_OSS_120B,
            ModelTeams.Remote.QWEN_235B,

            #ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_QWEN8B,

            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_30B,
            ModelTeams.Local.QWEN_CODER30B,

            ModelTeams.Local.GPT_OSS_20B,

        ],
        reruns=5,
    ),

    PhaseDefinition(
        phase="baseline-mix",
        configs=[
            CONFIGURATIONS["baseline-img"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            #Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            #Levels.DETAILED_INSTRUCT.TOMATO_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            #Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            #Levels.VAGUE_INSTRUCT.TOMATO_HARD,
            ],
        model_teams=[
            ModelTeams.Hybrid.DEEPSEEK_LLAMA,
            ModelTeams.Hybrid.MISTRAL_SMALL,
            ModelTeams.Hybrid.NEMOTRON_LLAMA,

            ModelTeams.Local.DOLPHIN_X_QWEN,
            ModelTeams.Local.PHI4_X_QWEN,
            ModelTeams.Local.NEMOTRON_NEMOTRON14B_X_QWEN,
        ],
        reruns=5,
    ),

    PhaseDefinition(
        phase="feature",
        configs=[
            CONFIGURATIONS["a-agent"],
            CONFIGURATIONS["b-agent"],
            CONFIGURATIONS["free-agent"],
            CONFIGURATIONS["step-agent"],
            CONFIGURATIONS["plan-agent"],
            ],
        levels=[

            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            Levels.DETAILED_INSTRUCT.TOMATO_HARD,
            Levels.DETAILED_INSTRUCT.POTATO_EASY,
            Levels.DETAILED_INSTRUCT.POTATO_HARD,
            Levels.DETAILED_INSTRUCT.ONION_EASY,
            Levels.DETAILED_INSTRUCT.ONION_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            Levels.VAGUE_INSTRUCT.TOMATO_HARD,
            Levels.VAGUE_INSTRUCT.POTATO_EASY,
            Levels.VAGUE_INSTRUCT.POTATO_HARD,
            Levels.VAGUE_INSTRUCT.ONION_EASY,
            Levels.VAGUE_INSTRUCT.ONION_HARD,
            ],
        model_teams=[
            ModelTeams.Remote.QWEN_235B,
            ModelTeams.Local.GPT_OSS_20B,
            ModelTeams.Local.QWEN_8B_OLLAMA,
        ],
        reruns=3,
    ),
)

# Map of hostname → runner configuration. Edit the keys to match your devices.
RUNNER_CONFIGS: Dict[str, RunnerConfig] = {
    # Laptop profile: keep both imaginator and realisator local/small.
    "r2d2xxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Hybrid.DEEPSEEK_LLAMA,
            ModelTeams.Hybrid.MISTRAL_SMALL,
            ModelTeams.Hybrid.NEMOTRON_LLAMA,

            ModelTeams.Remote.MAGISTRAL_SMALL,
            ModelTeams.Remote.GPT_OSS_120B,
            ModelTeams.Remote.QWEN_235B,

            # simple <8B
            ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_QWEN8B,

            ModelTeams.Local.QWEN_8B_LLAMA,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_LLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,
            ModelTeams.Local.QWEN_4B_Q4_LLAMA,
            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,
        ],
        allowed_phases=["backend", "baseline-solo", "baseline-mix"],
    ),
    # PC profile: allow remote imaginator with local realisator.
    "c3poxxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Local.DOLPHIN_X_QWEN,
            ModelTeams.Local.PHI4_X_QWEN,

            # simple <8B
            ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_QWEN8B,

            ModelTeams.Local.QWEN_8B_LLAMA,
            ModelTeams.Local.QWEN_4B_LLAMA,
            ModelTeams.Local.QWEN_4B_Q4_LLAMA,

            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,
            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,

            ModelTeams.Local.GPT_OSS_20B,
        ],
        allowed_phases=["backend", "baseline-solo", "baseline-mix"],
    ),
    "zedim-pc": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Local.QWEN_30B,
            ModelTeams.Local.QWEN_CODER30B,
            ModelTeams.Local.NEMOTRON_NEMOTRON14B_X_QWEN,

            ModelTeams.Local.GPT_OSS_20B,

            # simple <8B
            ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_QWEN8B,

            ModelTeams.Local.QWEN_8B_LLAMA,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_LLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,
            ModelTeams.Local.QWEN_4B_Q4_LLAMA,
            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,

            ModelTeams.Local.PHI4_X_QWEN,
            ModelTeams.Local.DOLPHIN_X_QWEN,
        ],
        allowed_phases=["backend", "baseline-solo", "baseline-mix"],
    )
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
