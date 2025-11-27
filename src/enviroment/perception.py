from __future__ import annotations
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING
import config
from config import PerceptionType

if TYPE_CHECKING:
    from enviroment.entity import Entity

class DetailLevel:
    NONE        = 0
    BAD         = 1
    REDUCED     = 3
    NORMAL      = 5
    GOOD        = 7
    EXTENDED    = 9
    FULL        = 10
    OMNISCIENT  = FULL + 1   # all information: drills down into every leaf of the tree

    def __init__(self, value: int) -> None:
        self.value = value

    def reduce(self) -> "DetailLevel":
        if self is DetailLevel.OMNISCIENT or self is DetailLevel.NONE:
            return self
        else:
            return DetailLevel(self.value - 1 * config.DEPTH_FAKTOR)

    def obfuscate_number(self, n: int) -> str:
        if self == DetailLevel.NONE:
            return "unknown"

        # zero case
        if n == 0:
            return "nothing"

        # only existence vs. non-existence
        if self == DetailLevel.BAD:
            return "some"

        # vague small vs. large distinction
        if self == DetailLevel.REDUCED:
            if n < 3:
                return "one or two"
            elif n < 6:
                return "a few"
            else:
                return "several"

        # small numbers fairly precise, large numbers vague
        if self == DetailLevel.NORMAL:
            if n <= 4:
                return str(n)
            elif n <= 10:
                return "a few"
            else:
                return "several"

        # gives rough approximate numbers, ±10%
        if self == DetailLevel.GOOD:
            delta = max(1, n // 10)
            noisy = n + random.randint(-delta, delta)
            return f"around {max(1, noisy)}"

        # gives a range, ±15%
        if self == DetailLevel.EXTENDED:
            tol = max(1, n // 7)
            low = max(0, n - tol)
            high = n + tol
            return f"between {low} and {high}"

        # almost exact, ±5%
        if self == DetailLevel.FULL:
            delta = max(1, n // 20)
            noisy = n + random.randint(-delta, delta)
            return f"about {max(1, noisy)}"

        # exact
        return str(n)

@dataclass
class PerceptionEnviroment:
    distance_m: float
    light_level: float = 1.0
    
class Datum:
    def __init__(self, key: str, value: str, info: dict[str, object], min_level: DetailLevel = DetailLevel.BAD) -> None:
        self.key = key
        self.value = value
        self.min_level = min_level
        self.info = info

    def perceive(self, observer: Entity, env: "PerceptionEnviroment", level: DetailLevel) -> None:
        perceived = False

        if config.ACTIVE_CONFIG.perception == PerceptionType.ALL:
            perceived = True

        if env.distance_m <= config.ACTIVE_CONFIG.perception_distance:
            perceived = True

        if level < self.min_level:
            perceived = False

        if perceived:
            self.info[self.key] = self.value
        else:
            self.info["object"] = "unknown"