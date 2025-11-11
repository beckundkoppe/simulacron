# config.py
from dataclasses import dataclass
from enum import Enum, auto

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
    PLAN   = auto(),
    REPLAN = auto(),

@dataclass(frozen=True)
class AgentConfiguration:
    imaginator: bool
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
    
ACTIVE_CONFIG: Configuration = None

PERCEPTION_DISTANCE: float = 4.0
INTERACTION_DISTANCE: float = 1.5

DEPTH_FAKTOR: int = 1
# faktor=1 => 10 depth steps visible
# faktor=2 => 5 depth steps visible