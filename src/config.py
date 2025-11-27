# config.py
from dataclasses import dataclass
from enum import Enum, auto

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

class PerceptionType(Enum):
    ALL = auto()
    DISTANCE = auto()

class PositionType(Enum):
    ROOMLESS = auto()
    RELATIVE = auto()

@dataclass(frozen=True)
class Configuration:
    agent: AgentConfiguration
    perception: PerceptionType
    position: PositionType
    temperature: float
    name: str
    seed: int = 12345
    perception_distance = 4.0
    interaction_distance = 1.5

    depth_factor = 1 # higher value would make perception harder: less deep discouveries into children of containers
    # factor=1 => 10 depth steps visible
    # factor=2 => 5 depth steps visible
    
ACTIVE_CONFIG: Configuration = None
APPEND_RAW = None


@dataclass(frozen=True)
class Backend:
    n_gpu_layers: int = -1
    n_threads: int = 24
    n_context: int = 8096