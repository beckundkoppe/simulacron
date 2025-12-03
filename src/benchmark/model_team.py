from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from llm.model import Model, Location


@dataclass(frozen=True)
class ModelTeam:
    """Pairing of models used for imagination and realisation."""

    realisator: Model
    imaginator: Model | None = None
    extra: Model | None = None
    tag: str | None = None

    def toObject(self):
        return {
            "name": self.getName(),
            "imaginator": self.imaginator.value.name if self.imaginator else "None",
            "realisator": self.realisator.value.name
        }
    
    def getName(self):
        if self.imaginator:
            return f"{self.imaginator.value.name}_+_{self.realisator.value.name}"
        return self.realisator.value.name

    def label(self) -> str:
        imaginator = self.imaginator or self.realisator
        if imaginator == self.realisator:
            return self.realisator.value.name
        return f"{self.realisator.value.name}+{imaginator.value.name}"

    def token(self) -> str:
        """Identifier used in filenames; prefers explicit tag; otherwise auto-build."""
        if self.tag:
            return self.tag

        real = self.realisator.value
        imag = (self.imaginator or self.realisator).value

        def _loc_suffix(spec) -> str:
            return "local" if spec.location is Location.LOCAL else "remote"

        real_name = real.name
        imag_name = imag.name

        # If same model, include one name and location.
        if self.imaginator is None or self.imaginator == self.realisator:
            return f"{real_name}-{_loc_suffix(real)}"

        # Different models → include both plus a composite location marker.
        location_tag = (
            "hybrid"
            if real.location is Location.LOCAL and imag.location is Location.REMOTE
            or real.location is Location.REMOTE and imag.location is Location.LOCAL
            else f"{_loc_suffix(real)}-{_loc_suffix(imag)}"
        )
        return f"{real_name}+{imag_name}-{location_tag}"

    def imaginator_or_default(self) -> Model:
        return self.imaginator or self.realisator

    def extra_or_default(self) -> Model:
        return self.extra or self.realisator


class ModelTeams(Enum):
    class Local(Enum):
        GROQ_LLAMA8B = ModelTeam(
            realisator=Model.Local.LlamaCpp.Llama3.LLAMA3_GROQ_8B_Q8,
        )

        NEMOTRON_QWEN8B = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.NEMOTRON_14B_Q8,
        )

        QWEN_4B = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.VANILLA_4B,
        )

        QWEN_8B_LLAMA = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.VANILLA_8B,
        )

        QWEN_8B_OLLAMA = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.VANILLA_8B,
        )

        QWEN_30B = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.VANILLA_30B,        
        )

        QWEN_CODER30B = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.CODER_30B,
        )

        GPT_OSS_20B = ModelTeam(
            realisator=Model.Local.Ollama.GPT_OSS_20B,
        )

        PHI4_X_QWEN = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.VANILLA_8B,
            imaginator=Model.Local.Ollama.PHI4_MINI_3_8B,
        )

        DOLPHIN_X_QWEN = ModelTeam(
            realisator=Model.Local.Ollama.Qwen3.VANILLA_8B,
            imaginator=Model.Local.Ollama.DOLPHIN3_8B,
        )

        NEMOTRON_NEMOTRON14B_X_QWEN8B = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.VANILLA_8B,
            imaginator=Model.Local.LlamaCpp.Qwen3.NEMOTRON_14B_Q8,
        )

# # # # # # # # # remote # # # # # # # # # # #
                           
    class Hybrid(Enum):
        MISTRAL_SMALL = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.VANILLA_8B,
            imaginator=Model.Remote.MISTRAL_SMALL_24B,
        )

        NEMOTRON_LLAMA = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.VANILLA_8B,
            imaginator=Model.Remote.NEMOTRON_SUPER_49B,
        )

        DEEPSEEK_LLAMA = ModelTeam(
            realisator=Model.Local.LlamaCpp.Qwen3.VANILLA_8B,
            imaginator=Model.Remote.DEEPSEEK_R1_LLAMA_DISTILL_70B,
        )

    class Remote(Enum):
        GPT_OSS_120B = ModelTeam(
            realisator=Model.Remote.GPT_OSS_120B,
        )

        QWEN_235B = ModelTeam(
            realisator=Model.Remote.QWEN3_235B,
        )

        MAGISTRAL_SMALL = ModelTeam(
            realisator=Model.Remote.MAGISTRAL_SMALL_24B,
        )

MODEL_TEAM_PRESETS: Sequence[ModelTeams] = tuple(ModelTeams)
