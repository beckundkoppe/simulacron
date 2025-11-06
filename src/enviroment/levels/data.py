from dataclasses import dataclass
from typing import Callable

from enviroment.entity import Entity

@dataclass()
class LevelSpec:
    agent_entities: list[tuple[Entity, str]]
    is_success: Callable[[], bool]