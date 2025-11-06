# config.py
from dataclasses import dataclass
from enum import Enum, auto

class PlanType(Enum):
    OFF     = auto(),
    PLAN    = auto(),
    REPLAN  = auto(),

class PerceptionType(Enum):
    FULL        = auto(),
    DISTANCE    = auto(),
    SENSE       = auto(),

class PositionType(Enum):
    ROOMLESS    = auto(),
    CHESSBOARD  = auto(),
    RELATIVE    = auto(),

@dataclass(frozen=True)
class Configuration:
    perception: PerceptionType
    position_type: PositionType
    imagine_feature: bool
    learning_feature: bool
    planning_feature: PlanType
    temperature: float
    name: str
    
CONFIG = None

DISTANCE = 4.0