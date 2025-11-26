# config.py
from dataclasses import dataclass
from enum import Enum, auto

class ImaginatorType(Enum):
    OFF = auto()
    ON = auto()
    QUESTION = auto()
    MULTIPLE = auto()

class ObserveType(Enum):
    OFF = auto()
    ON = auto()
    MEMORIZE = auto()

class ReflectType(Enum):
    OFF = auto()
    ON = auto()
    MEMORIZE = auto()

class PlanType(Enum):
    OFF = auto()
    FREE = auto()
    STEP = auto()
    DECOMPOSE = auto()
    TRY = auto()

@dataclass(frozen=True)
class AgentConfiguration:
    imaginator: ImaginatorType
    observe: ObserveType
    reflect: ReflectType
    plan: PlanType

class PerceptionType(Enum):
    ALL = auto()
    DISTANCE = auto()

class PositionType(Enum):
    ROOMLESS = auto()
    CHESSBOARD = auto()
    RELATIVE = auto()

@dataclass(frozen=True)
class Configuration:
    perception: PerceptionType
    position: PositionType
    agent: AgentConfiguration
    temperature: float
    name: str
    seed: int = 12345
    
ACTIVE_CONFIG: Configuration = None
APPEND_RAW = None


PERCEPTION_DISTANCE: float = 4.0
INTERACTION_DISTANCE: float = 1.5

DEPTH_FAKTOR: int = 1
# faktor=1 => 10 depth steps visible
# faktor=2 => 5 depth steps visible

@dataclass(frozen=True)
class Backend:
    n_gpu_layers: int = -1
    n_threads: int = 24
    n_context: int = 8096

    @classmethod
    def effective_gpu_layers(cls) -> int:
        return cls.n_gpu_layers

    @classmethod
    def effective_threads(cls) -> int:
        return cls.n_threads

    @classmethod
    def effective_context(cls) -> int:
        return cls.n_context
