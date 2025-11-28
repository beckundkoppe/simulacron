from dataclasses import dataclass

from config import Configuration
from enviroment.levels.level import Level
from llm.model import Model


@dataclass(frozen=True)
class Run:
    configuration: Configuration
    main_model: Model
    level: Level
    reruns: int
    optimal_steps_multiplier: float
    imaginator: Model | None = None
    extra_model: Model | None = None
