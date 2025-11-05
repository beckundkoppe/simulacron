from dataclasses import dataclass
from typing import Dict, List, TypedDict

from config import Configuration, PerceptionType, PlanType, PositionType
from llm.model import Model

BASE_TEMP = 0.0

def baseline() -> List[Configuration]:
    return [
        Configuration(PerceptionType.FULL,     PositionType.ROOMLESS, False, False, PlanType.OFF, BASE_TEMP, "baseline-roomless-full"),

        Configuration(PerceptionType.FULL,     PositionType.RELATIVE, False, False, PlanType.OFF, BASE_TEMP, "baseline-relative-full"),
        Configuration(PerceptionType.DISTANCE, PositionType.RELATIVE, False, False, PlanType.OFF, BASE_TEMP, "baseline-relative-distance"),
        Configuration(PerceptionType.SENSE,    PositionType.RELATIVE, False, False, PlanType.OFF, BASE_TEMP, "baseline-relative-sense"),

        Configuration(PerceptionType.FULL,     PositionType.CHESSBOARD, False, False, PlanType.OFF, BASE_TEMP, "baseline-chessboard-full"),
        Configuration(PerceptionType.DISTANCE, PositionType.CHESSBOARD, False, False, PlanType.OFF, BASE_TEMP, "baseline-chessboard-distance"),
        Configuration(PerceptionType.SENSE,    PositionType.CHESSBOARD, False, False, PlanType.OFF, BASE_TEMP, "baseline-chessboard-sense"),
    ]

def feature_imagine() -> List[Configuration]:
    configs: List[Configuration] = []

    configs.extend([
        Configuration(PerceptionType.FULL,  PositionType.CHESSBOARD, True, False, PlanType.OFF, BASE_TEMP, "imagine-full"),
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, False, PlanType.OFF, BASE_TEMP, "imagine-partial"),
    ])

    return configs

def feature_learning() -> List[Configuration]:
    configs: List[Configuration] = []

    configs.extend([
        Configuration(PerceptionType.FULL,  PositionType.CHESSBOARD, False, True, PlanType.OFF, BASE_TEMP, "feature-learning-full"),
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, False, True, PlanType.OFF, BASE_TEMP, "feature-learning-partial"),
    ])

    return configs

def feature_planning() -> List[Configuration]:
    configs: List[Configuration] = []

    configs.extend([
        Configuration(PerceptionType.FULL,  PositionType.CHESSBOARD, False, False, PlanType.PLAN, BASE_TEMP,   "feature-planning-full"),
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, False, False, PlanType.PLAN, BASE_TEMP,   "feature-planning-partial"),
        Configuration(PerceptionType.FULL,  PositionType.CHESSBOARD, False, False, PlanType.REPLAN, BASE_TEMP, "feature-replanning-full"),
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, False, False, PlanType.REPLAN, BASE_TEMP, "feature-replanning-partial"),
    ])

    return configs

def combine() -> List[Configuration]:
    configs: List[Configuration] = []

    configs.extend([
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, True, PlanType.OFF,    BASE_TEMP, "combo-il"),
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, True, PlanType.REPLAN, BASE_TEMP, "combo-ilp"),
    ])

    return configs

def temps() -> List[Configuration]:
    temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    configs: List[Configuration] = []

    for t in temps:
        configs.extend([
            Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, True, PlanType.REPLAN, t, f"temps-{t}"),
        ])
    return configs

def models() -> List[Configuration]:
    configs: List[Configuration] = []

    configs.extend([
        Configuration(PerceptionType.SENSE, PositionType.CHESSBOARD, True, True, PlanType.REPLAN, BASE_TEMP, "models-full"),
    ])

    return configs

MODELS_REDUCED = [
        #lokal
        Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B,

        #institut
        Model.Remote.GPT_OSS_20B,

        #kommerziell
        Model.Remote.GPT5
    ]

MODELS_ALL = [
        #lokal
        Model.Local.Ollama.HYBRID_LLAMA3_1_NEMOTRON_8B,
        Model.Local.LlamaCpp.HYBRID_DEEPSEEK_QWEN_7B,
        Model.Local.LlamaCpp.HYBRID_PHI4_MINI_3_8B,

        #institut
        Model.Remote.PHI4_PLUS,
        Model.Remote.GPT_OSS_20B,

        #kommerziell
        Model.Remote.GPT4,
        Model.Remote.GPT5,
    ]

class ConfigurationSet(TypedDict):
    configs: List[Configuration]
    models: List["Model"]

def get_sets() -> Dict[str, ConfigurationSet]:
    return {
        "baseline": {
            "configs": baseline(),
            "models": MODELS_REDUCED,
        },

        "feature_imagine": {
            "configs": feature_imagine(),
            "models": MODELS_REDUCED,
        },
        "feature_planning": {
            "configs": feature_planning(),
            "models": MODELS_REDUCED,
        },
        "feature_learning": {
            "configs": feature_learning(),
            "models": MODELS_REDUCED,
        },
        "combine": {
            "configs": combine(),
            "models": MODELS_ALL,
        },

        "temps": {
            "configs": temps(),
            "models": MODELS_REDUCED,
        },
        "models": {
            "configs": models(),
            "models": MODELS_ALL,
        },
    }