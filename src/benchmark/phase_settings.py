from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        phase="solo-baseline-easy",
        configs=[
            CONFIGURATIONS["baseline-naive"],
            CONFIGURATIONS["baseline-img"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            Levels.DETAILED_INSTRUCT.PEPPER_EASY,
            Levels.DETAILED_INSTRUCT.PEPPER_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            Levels.VAGUE_INSTRUCT.PEPPER_EASY,
            Levels.VAGUE_INSTRUCT.PEPPER_HARD,

            ],
        model_teams=[
            #ModelTeams.Remote.MAGISTRAL_SMALL, #TODO
            ModelTeams.Remote.GPT_OSS_120B,
            ModelTeams.Remote.QWEN_235B,

            #ModelTeams.Local.GROQ_LLAMA8B, ### MANUELLER ABBRUCH
            #ModelTeams.Local.NEMOTRON_8B, #TODO

            #ModelTeams.Local.QWEN_4B_Q4_OLLAMA, #TODO
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_30B,
            ModelTeams.Local.QWEN_CODER30B,

            ModelTeams.Local.GPT_OSS_20B,

        ],
        reruns=5,
    ),

    PhaseDefinition(
        phase="mix-baseline-easy",
        configs=[
            CONFIGURATIONS["baseline-img"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            Levels.DETAILED_INSTRUCT.PEPPER_EASY,
            Levels.DETAILED_INSTRUCT.PEPPER_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            Levels.VAGUE_INSTRUCT.PEPPER_EASY,
            Levels.VAGUE_INSTRUCT.PEPPER_HARD,
            ],
        model_teams=[
            #ModelTeams.Hybrid.DEEPSEEK_LLAMA,
            #ModelTeams.Hybrid.MISTRAL_SMALL,
            #ModelTeams.Hybrid.NEMOTRON_LLAMA,

            #ModelTeams.Local.DOLPHIN_X_QWEN,
            #ModelTeams.Local.PHI4_X_QWEN,
            #ModelTeams.Local.NEMOTRON_NEMOTRON14B_X_QWEN, ### MANUELLER ABBRUCH
        ],
        reruns=5,
    ),

    PhaseDefinition(
        phase="feature-easy",
        configs=[
            #CONFIGURATIONS["baseline-retry"],
            CONFIGURATIONS["baseline-naive"],
            CONFIGURATIONS["baseline-img"],
            CONFIGURATIONS["imginator-retry"],
            #CONFIGURATIONS["imginator-question"],
            #CONFIGURATIONS["a-agent"],
            #CONFIGURATIONS["b-agent"],
            CONFIGURATIONS["step-agent"],
            CONFIGURATIONS["tree-agent"],
            CONFIGURATIONS["trial-agent"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.CARROT_EASY,
            Levels.DETAILED_INSTRUCT.CARROT_HARD,
            Levels.DETAILED_INSTRUCT.CUCUMBER_EASY,
            Levels.DETAILED_INSTRUCT.CUCUMBER_HARD,
            Levels.DETAILED_INSTRUCT.PEPPER_EASY,
            Levels.DETAILED_INSTRUCT.PEPPER_HARD,
            Levels.DETAILED_INSTRUCT.SALAD_EASY,
            Levels.DETAILED_INSTRUCT.SALAD_HARD,

            Levels.VAGUE_INSTRUCT.CARROT_EASY,
            Levels.VAGUE_INSTRUCT.CARROT_HARD,
            Levels.VAGUE_INSTRUCT.CUCUMBER_EASY,
            Levels.VAGUE_INSTRUCT.CUCUMBER_HARD,
            Levels.VAGUE_INSTRUCT.PEPPER_EASY,
            Levels.VAGUE_INSTRUCT.PEPPER_HARD,
            Levels.VAGUE_INSTRUCT.SALAD_EASY,
            Levels.VAGUE_INSTRUCT.SALAD_HARD,
            ],
        model_teams=[
            ModelTeams.Remote.GPT_OSS_120B,
            #ModelTeams.Remote.QWEN_235B,

            #ModelTeams.Local.QWEN_4B_Q4_OLLAMA, #TODO
            ModelTeams.Local.QWEN_8B_OLLAMA,
            #ModelTeams.Local.QWEN_30B,
            ModelTeams.Local.QWEN_CODER30B,

            ModelTeams.Local.GPT_OSS_20B,
        ],
        reruns=2,
    ),

    PhaseDefinition(
        phase="feature-hard",
        configs=[
            #CONFIGURATIONS["baseline-naive"], #TODO
            #CONFIGURATIONS["baseline-img"], #TODO
            #CONFIGURATIONS["imginator-retry"], #TODO
            #CONFIGURATIONS["a-agent"], #TODO
            #CONFIGURATIONS["b-agent"], #TODO
            CONFIGURATIONS["step-agent"],
            CONFIGURATIONS["tree-agent"],
            CONFIGURATIONS["trial-agent"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            #Levels.DETAILED_INSTRUCT.TOMATO_HARD,
            Levels.DETAILED_INSTRUCT.POTATO_EASY,
            #Levels.DETAILED_INSTRUCT.POTATO_HARD,
            Levels.DETAILED_INSTRUCT.ONION_EASY,
            #Levels.DETAILED_INSTRUCT.ONION_HARD,

            Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            #Levels.VAGUE_INSTRUCT.TOMATO_HARD,
            Levels.VAGUE_INSTRUCT.POTATO_EASY,
            #Levels.VAGUE_INSTRUCT.POTATO_HARD,
            Levels.VAGUE_INSTRUCT.ONION_EASY,
            #Levels.VAGUE_INSTRUCT.ONION_HARD,
            ],
        model_teams=[
            #ModelTeams.Remote.QWEN_235B,
            ModelTeams.Remote.GPT_OSS_120B,

            #ModelTeams.Local.GPT_OSS_20B,
            ModelTeams.Local.QWEN_CODER30B,

            ModelTeams.Local.QWEN_8B_OLLAMA,
            #ModelTeams.Local.QWEN_4B_Q4_OLLAMA,
        ],
        reruns=2,
    ),

    PhaseDefinition(
        phase="baseline-hard",
        configs=[
            CONFIGURATIONS["baseline-naive"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            Levels.DETAILED_INSTRUCT.TOMATO_HARD,
            Levels.DETAILED_INSTRUCT.POTATO_EASY,
            Levels.DETAILED_INSTRUCT.POTATO_HARD,
            Levels.DETAILED_INSTRUCT.ONION_EASY,
            Levels.DETAILED_INSTRUCT.ONION_HARD,

            Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            Levels.VAGUE_INSTRUCT.TOMATO_HARD,
            Levels.VAGUE_INSTRUCT.POTATO_EASY,
            Levels.VAGUE_INSTRUCT.POTATO_HARD,
            Levels.VAGUE_INSTRUCT.ONION_EASY,
            Levels.VAGUE_INSTRUCT.ONION_HARD,
            ],
        model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
        ],
        reruns=2,
    ),

    PhaseDefinition(
        phase="selected-feature-hard",
        configs=[
            CONFIGURATIONS["baseline-naive"],
            CONFIGURATIONS["baseline-img"],
            CONFIGURATIONS["imginator-retry"],
            CONFIGURATIONS["trial-agent"],
            #CONFIGURATIONS["lea"],
            #CONFIGURATIONS["lea-explore"],
            ],
        levels=[
            Levels.DETAILED_INSTRUCT.TOMATO_EASY,
            Levels.DETAILED_INSTRUCT.TOMATO_HARD,
            Levels.DETAILED_INSTRUCT.POTATO_EASY,
            Levels.DETAILED_INSTRUCT.POTATO_HARD,
            Levels.DETAILED_INSTRUCT.ONION_EASY,
            Levels.DETAILED_INSTRUCT.ONION_HARD,

            Levels.VAGUE_INSTRUCT.TOMATO_EASY,
            Levels.VAGUE_INSTRUCT.TOMATO_HARD,
            Levels.VAGUE_INSTRUCT.POTATO_EASY,
            Levels.VAGUE_INSTRUCT.POTATO_HARD,
            Levels.VAGUE_INSTRUCT.ONION_EASY,
            Levels.VAGUE_INSTRUCT.ONION_HARD,
            ],
        model_teams=[
            ModelTeams.Remote.GPT5_NANO,
        ],
        reruns=2,
    ),

    PhaseDefinition(
        phase="supermem",
        configs=[
            #CONFIGURATIONS["baseline-img"], best
            ],
        levels=[
            #Levels.DETAILED_INSTRUCT.PEPPER_EASY,
            #Levels.DETAILED_INSTRUCT.PEPPER_HARD,

            #Levels.VAGUE_INSTRUCT.PEPPER_EASY,
            #Levels.VAGUE_INSTRUCT.PEPPER_HARD,
            ],
        model_teams=[
            #ModelTeams.Remote.MAGISTRAL_SMALL,
            #ModelTeams.Local.QWEN_CODER30B,
            #ModelTeams.Local.QWEN_8B_OLLAMA,
        ],
        reruns=2,
    ),

    PhaseDefinition(
        phase="temperature",
        configs=[
            #CONFIGURATIONS["baseline-naive"],
            #CONFIGURATIONS["best-02"],
            #CONFIGURATIONS["best-04"],
            #CONFIGURATIONS["best-06"],
            #CONFIGURATIONS["best-08"],
            #CONFIGURATIONS["best-10"],
            ],
        levels=[
            #Levels.VAGUE_INSTRUCT.CARROT_EASY,
            #Levels.VAGUE_INSTRUCT.SALAD_EASY,
            #Levels.VAGUE_INSTRUCT.POTATO_EASY,
            ],
        model_teams=[
            #ModelTeams.Remote.QWEN_235B,
            #ModelTeams.Local.QWEN_CODER30B,
            #ModelTeams.Local.QWEN_8B_OLLAMA,
        ],
        reruns=10,
    ),
)

# Map of hostname → runner configuration. Edit the keys to match your devices.
RUNNER_CONFIGS: Dict[str, RunnerConfig] = {
    # Laptop profile: keep both imaginator and realisator local/small.
    "r2d2xxx": RunnerConfig(
        allowed_model_teams=[
            #ModelTeams.Hybrid.DEEPSEEK_LLAMA,
            #ModelTeams.Hybrid.MISTRAL_SMALL,
            #ModelTeams.Hybrid.NEMOTRON_LLAMA,

            #ModelTeams.Remote.MAGISTRAL_SMALL,
            #ModelTeams.Remote.GPT_OSS_120B,
            #ModelTeams.Remote.QWEN_235B,

            ## simple <8B
            #ModelTeams.Local.GROQ_LLAMA8B,
            #ModelTeams.Local.NEMOTRON_8B,
#
            #ModelTeams.Local.QWEN_8B_LLAMA,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            #ModelTeams.Local.QWEN_4B_LLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,
            #ModelTeams.Local.QWEN_4B_Q4_LLAMA,
            ModelTeams.Local.QWEN_4B_Q4_OLLAMA,
#
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "bb8xxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Hybrid.DEEPSEEK_LLAMA,
            ModelTeams.Hybrid.MISTRAL_SMALL,
            ModelTeams.Hybrid.NEMOTRON_LLAMA,

            ModelTeams.Remote.MAGISTRAL_SMALL,
            ModelTeams.Remote.GPT_OSS_120B,
            #ModelTeams.Remote.QWEN_235B,

            ## simple <8B
            #ModelTeams.Local.GROQ_LLAMA8B,
            #ModelTeams.Local.NEMOTRON_8B,

            #ModelTeams.Local.QWEN_8B_LLAMA,
            #ModelTeams.Local.QWEN_8B_OLLAMA,
            #ModelTeams.Local.QWEN_4B_LLAMA,
            #ModelTeams.Local.QWEN_4B_OLLAMA,
            #ModelTeams.Local.QWEN_4B_Q4_LLAMA,
            #ModelTeams.Local.QWEN_4B_Q4_OLLAMA,

            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    # PC profile: allow remote imaginator with local realisator.
    "c3poxxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Local.DOLPHIN_X_QWEN,
            ModelTeams.Local.PHI4_X_QWEN,

            # simple <8B
            ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_NEMOTRON14B_X_QWEN,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,

            ModelTeams.Local.GPT_OSS_20B,

            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "gonkxxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "k2soxxx": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "zedim-pc": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Local.QWEN_30B,
            ModelTeams.Local.QWEN_CODER30B,
            ModelTeams.Local.NEMOTRON_NEMOTRON14B_X_QWEN,

            ModelTeams.Local.GPT_OSS_20B,

            # simple <8B
            ModelTeams.Local.GROQ_LLAMA8B,
            ModelTeams.Local.NEMOTRON_8B,
            ModelTeams.Local.QWEN_8B_OLLAMA,
            ModelTeams.Local.QWEN_4B_OLLAMA,

            ModelTeams.Local.PHI4_X_QWEN,
            ModelTeams.Local.DOLPHIN_X_QWEN,

            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "openai-runner-1": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "openai-runner-2": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "openai-runner-3": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "openai-runner-4": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
    "openai-runner-5": RunnerConfig(
        allowed_model_teams=[
            ModelTeams.Remote.GPT5_NANO,
            ModelTeams.Remote.GPT5_MINI,
            ModelTeams.Remote.GPT5_1,
            ModelTeams.Remote.GPT4_1,
            ModelTeams.Remote.GPTo3,
        ],
        allowed_phases=[
            "backend",
            "solo-baseline-easy",
            "mix-baseline-easy",
            "feature-easy",
            "feature-hard",
            "baseline-hard",
            "selected-feature-hard",
            "supermem",
            "temperature",
        ],
    ),
}

# Convenience defaults so other modules can import a single definition.
PHASE_DEFINITION = PHASES[0]
DEFAULT_RUNNER_CONFIG = RunnerConfig(
    allowed_model_teams=PHASE_DEFINITION.model_teams, allowed_phases=[PHASE_DEFINITION.phase]
)

_RUNNER_HOSTNAME_FILE = Path(__file__).resolve().parent.parent.parent / "runner_hostname.txt"


def get_runner_hostname() -> str:
    """
    Return the runner hostname, preferring an override file in the repo root.
    """
    if _RUNNER_HOSTNAME_FILE.exists():
        override = _RUNNER_HOSTNAME_FILE.read_text().strip()
        if override:
            return override
    return gethostname()


def resolve_runner_config(hostname: str | None = None, phase: str | None = None) -> ResolvedRunnerConfig:
    """Return the runner config for the given hostname (optional phase override)."""

    host = hostname or get_runner_hostname()
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
