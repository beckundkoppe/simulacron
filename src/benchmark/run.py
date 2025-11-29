from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.model_team import ModelTeam
from config import Configuration
from enviroment.levels.level import Level

if TYPE_CHECKING:
    from llm.model import Model


@dataclass(frozen=True)
class Run:
    configuration: Configuration
    model_team: ModelTeam
    level: Level
    reruns: int
    optimal_steps_multiplier: float

    @property
    def main_model(self) -> "Model":
        return self.model_team.realisator

    @property
    def imaginator(self) -> "Model":
        return self.model_team.imaginator_or_default()

    @property
    def extra_model(self) -> "Model":
        return self.model_team.extra_or_default()
