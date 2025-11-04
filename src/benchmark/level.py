from dataclasses import dataclass
from enum import Enum

class LevelKind(Enum):
    SHORT           = "short horizon"
    LONG            = "long horizon"

@dataclass
class LevelSpec:
    name: str
    kind: LevelKind

class Level(Enum):
    class Short(Enum):
        POTATO = LevelSpec(
                    name="Find potato",
                    kind=LevelKind.SHORT,
                )
        
        EXIT_SMALL = LevelSpec(
                    name="Exit small building",
                    kind=LevelKind.SHORT,
                )
    class Long(Enum):
        EXIT_BIG = LevelSpec(
                    name="Exit big building",
                    kind=LevelKind.LONG,
                )