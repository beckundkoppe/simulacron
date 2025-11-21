# config.py
from dataclasses import dataclass
from enum import Enum, auto

class ImaginatorType(Enum):
    OFF      = auto(),
    ON       = auto(),
    QUESTION = auto(),
    MULTIPLE = auto(),

class ObserveType(Enum):
    OFF      = auto(),
    ON       = auto(),
    MEMORIZE = auto(),

class ReflectType(Enum):
    OFF   = auto(),
    ON    = auto(),
    LEARN = auto(),

class PlanType(Enum):
    OFF    = auto(),
    FREE   = auto(),
    STRUCTURED = auto(),

@dataclass(frozen=True)
class AgentConfiguration:
    imaginator: ImaginatorType
    observe: ObserveType
    reflect: ReflectType
    plan: PlanType

class PerceptionType(Enum):
    ALL        = auto(),
    DISTANCE    = auto(),

class PositionType(Enum):
    ROOMLESS    = auto(),
    CHESSBOARD  = auto(),
    RELATIVE    = auto(),

@dataclass(frozen=True)
class Configuration:
    perception: PerceptionType
    position: PositionType
    agent: AgentConfiguration
    temperature: float
    name: str
    seed: int = 12345
    
ACTIVE_CONFIG: Configuration = None

PERCEPTION_DISTANCE: float = 4.0
INTERACTION_DISTANCE: float = 1.5

DEPTH_FAKTOR: int = 1
# faktor=1 => 10 depth steps visible
# faktor=2 => 5 depth steps visible

@dataclass(frozen=True)
class Backend:
    _n_gpu_layers = -1
    _n_threads = 24
    _n_context = 8096