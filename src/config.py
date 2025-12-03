# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

class MemoryType(str, Enum):
    SIMPLE = "simple"
    SUPER = "super"

class PlanType(Enum):
    OFF = auto()            # no plan step
    FREE = auto()           # freeform planning
    STEP = auto()           # structured step planning
    DECOMPOSE = auto()      # structured tree planning


class TrialType(Enum):
    ON = auto()             # no try step
    OFF = auto()            #

class ActionType(Enum):
    DIRECT = auto()         # no extra step just realisation
    IMAGINATOR = auto()     # imaginator -> realisator
    IMG_RETRY = auto()      # += realisator has 3 retries
    IMG_QUESTION = auto()   # += img-real cycle has 3 retries


class ObserveType(Enum):
    OFF = auto()            # off
    ON = auto()             # observation step
    MEMORIZE_ZERO = auto()  # += memorize important info
    MEMORIZE_FEW = auto()   # += examples what would be suficient


class ReflectType(Enum):
    OFF = auto()            # off
    ON = auto()             # reflection step
    MEMORIZE_ZERO = auto()  # += memorize important info
    MEMORIZE_FEW = auto()   # += examples what would be suficient

@dataclass(frozen=True)
class AgentConfiguration:
    plan: PlanType
    trial: TrialType
    action: ActionType
    observe: ObserveType
    reflect: ReflectType
    memory_type: MemoryType


class PerceptionType(Enum):
    ALL = auto()
    DISTANCE = auto()


class PositionType(Enum):
    ROOMLESS = auto()
    RELATIVE = auto()


@dataclass(frozen=True)
class Configuration:
    agents: AgentConfiguration
    perception: PerceptionType
    position: PositionType
    temperature: float
    name: str
    seed: int = 12345
    perception_distance = 4.0
    interaction_distance = 1.5

    # Higher values make perception harder by reducing visible depth into
    # container hierarchies. factor=1 => 10 depth steps visible, factor=2 =>
    # 5 depth steps visible.
    depth_factor = 1


DEFAULT_AGENT_CONFIGURATION = AgentConfiguration(
    plan=PlanType.OFF,
    trial=TrialType.OFF,
    action=ActionType.DIRECT,
    observe=ObserveType.OFF,
    reflect=ReflectType.OFF,
    memory_type=None,
)

# A sensible default configuration so environment logic can run outside of the
# benchmarking harness (e.g. unit tests) without having to bootstrap a full
# benchmark run first.
DEFAULT_CONFIGURATION = Configuration(
    agents=DEFAULT_AGENT_CONFIGURATION,
    perception=PerceptionType.ALL,
    position=PositionType.RELATIVE,
    temperature=0.0,
    name="default",
)

# The active runtime configuration. Components should update this when running
# with custom parameters (see benchmark.dispatcher), but a default is provided
# to keep environment interactions functional in isolation.
ACTIVE_CONFIG: Configuration = DEFAULT_CONFIGURATION

# Flag indicating whether raw model output should be appended to logs.
APPEND_RAW = None


@dataclass(frozen=True)
class Backend:
    n_gpu_layers: int = -1
    n_threads: int = 24
    n_context: int = 16384
