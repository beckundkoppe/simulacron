from dataclasses import dataclass
from enum import Enum
from typing import Callable

from enviroment.entity import Entity
from enviroment.levels.data import LevelSpec
from enviroment.levels.potato import build_easy, build_hard

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
    POTATO_EASY = Level("Potato (easy)", LevelType.SHORT, 8, build_easy)
    POTATO_HARD = Level("Potato (hard)", LevelType.SHORT, 8, build_hard)