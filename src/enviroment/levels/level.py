from dataclasses import dataclass
from enum import Enum
from typing import Callable

from enviroment.levels.data import LevelSpec
from enviroment.levels import carrot, cucumber, salad, tomato, potato, onion

class LevelType(Enum):
    SHORT           = "short horizon"
    LONG            = "long horizon"

@dataclass()
class Level:
    name: str
    easy: bool
    detailed: bool
    optimal_steps: int
    build: Callable[[bool], LevelSpec]

    def getName(self) -> str:
        dif = "easy" if self.easy else "hard"
        det = "detailed" if self.detailed else "vague"

        return self.name + "-" + dif + "-" + det
    
    def toObject(self):
        return {
            "name": self.getName(),
            "difficulty": "easy" if self.easy else "hard",
            "accuracy": "detailed" if self.detailed else "vague",
            "optimal_steps": self.optimal_steps,

        }

class Levels(Enum):
    class DETAILED_INSTRUCT(Enum):
        CARROT_EASY = Level("carrot", True, True, 1, lambda d: carrot.build_easy(d))
        CARROT_HARD = Level("carrot", False, True, 1, lambda d: carrot.build_hard(d))

        CUCUMBER_EASY = Level("cucumber", True, True, 3, lambda d: cucumber.build_easy(d))
        CUCUMBER_HARD = Level("cucumber", False, True, 3, lambda d: cucumber.build_hard(d))

        SALAD_EASY = Level("salad", True, True, 24, lambda d: salad.build_easy(d))
        SALAD_HARD = Level("salad", False, True, 24, lambda d: salad.build_hard(d))

        TOMATO_EASY = Level("tomato", True, True, 10, lambda d: tomato.build_easy(d))
        TOMATO_HARD = Level("tomato", False, True, 10, lambda d: tomato.build_hard(d))

        POTATO_EASY = Level("potato", True, True, 8, lambda d: potato.build_easy(d))
        POTATO_HARD = Level("potato", False, True, 8, lambda d: potato.build_medium(d))

        ONION_EASY = Level("onion", True, True, 10, lambda d: onion.build_easy(d))
        ONION_HARD = Level("onion", False, True, 10, lambda d: onion.build_hard(d))

    class VAGUE_INSTRUCT(Enum):
        CARROT_EASY = Level("carrot", True, False, 1, lambda d: carrot.build_easy(d))
        CARROT_HARD = Level("carrot", False, False, 1, lambda d: carrot.build_hard(d))

        CUCUMBER_EASY = Level("cucumber", True, False, 3, lambda d: cucumber.build_easy(d))
        CUCUMBER_HARD = Level("cucumber", False, False, 3, lambda d: cucumber.build_hard(d))

        SALAD_EASY = Level("salad", True, False, 24, lambda d: salad.build_easy(d))
        SALAD_HARD = Level("salad", False, False, 24, lambda d: salad.build_hard(d))

        TOMATO_EASY = Level("tomato", True, False, 10, lambda d: tomato.build_easy(d))
        TOMATO_HARD = Level("tomato", False, False, 10, lambda d: tomato.build_hard(d))

        POTATO_EASY = Level("potato", True, False, 8, lambda d: potato.build_easy(d))
        POTATO_HARD = Level("potato", False, False, 8, lambda d: potato.build_medium(d))

        ONION_EASY = Level("onion", True, False, 10, lambda d: onion.build_easy(d))
        ONION_HARD = Level("onion", False, False, 10, lambda d: onion.build_hard(d))
