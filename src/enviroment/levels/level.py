from dataclasses import dataclass
from enum import Enum
from typing import Callable

from enviroment.entity import Entity
from enviroment.levels.data import LevelSpec
from enviroment.levels.potato import build_easy, build_hard, build_medium

class LevelType(Enum):
    SHORT           = "short horizon"
    LONG            = "long horizon"

@dataclass()
class Level:
    name: str
    type: LevelType
    optimal_steps: int
    build: Callable[[], LevelSpec]

class Levels(Enum):
    class DETAILED_INSTRUCT(Enum):
        POTATO_EASY = Level("Potato (easy)", LevelType.SHORT, 8, lambda: build_easy(True))
        POTATO_MEDIUM = Level("Potato (medium)", LevelType.SHORT, 8, lambda: build_medium(True))
        POTATO_HARD = Level("Potato (hard)", LevelType.SHORT, 8, lambda: build_hard(True))

    class VAGUE_INSTRUCT(Enum):
        POTATO_EASY = Level("Potato (easy)", LevelType.SHORT, 8, lambda: build_easy(False))
        POTATO_MEDIUM = Level("Potato (medium)", LevelType.SHORT, 8, lambda: build_medium(False))
        POTATO_HARD = Level("Potato (hard)", LevelType.SHORT, 8, lambda: build_hard(False))