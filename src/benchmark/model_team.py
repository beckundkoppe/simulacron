from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from llm.model import Model


@dataclass(frozen=True)
class ModelTeam:
    """Pairing of models used for imagination and realisation."""

    realisator: Model
    imaginator: Model | None = None
    extra: Model | None = None

    def label(self) -> str:
        imaginator = self.imaginator or self.realisator
        if imaginator == self.realisator:
            return self.realisator.value.name
        return f"{self.realisator.value.name}+{imaginator.value.name}"

    def imaginator_or_default(self) -> Model:
        return self.imaginator or self.realisator

    def extra_or_default(self) -> Model:
        return self.extra or self.realisator


class ModelTeams(Enum):
    """Curated model team presets for benchmarks."""

    LOCAL_QWEN_CODER = ModelTeam(
        realisator=Model.Local.Ollama.Qwen3.CODER_30B,
    )
    HYBRID_MISTRAL_SMALL = ModelTeam(
        realisator=Model.Local.LlamaCpp.Qwen.VANILLA_8B,
        imaginator=Model.Remote.MISTRAL_SMALL_24B,
    )


MODEL_TEAM_PRESETS: Sequence[ModelTeams] = tuple(ModelTeams)
