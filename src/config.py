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
    
CONFIG = Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, True, PlanType.REPLAN, 0.0, "models-full")
