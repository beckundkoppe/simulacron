"""Learning management for dynamic and post-episode meta learnings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.benchresult import RunResult
    from config import Configuration
    from llm.memory.memory import Memory
    from llm.model import Model


class LearningEvent(str, Enum):
    """Categories for learnings generated during an episode."""

    FORMAL_ERROR = "formal_error"
    ACTION_FAILURE = "action_failure"
    ACTION_SUCCESS = "action_success"


class LearningType(str, Enum):
    """Persistence stage for a learning."""

    DYNAMIC = "dynamic"
    POST_EPISODE = "post_episode"


@dataclass
class Learning:
    """Single learning item with metadata."""

    type: LearningType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeContext:
    """Aggregates learnings for the active episode."""

    level_name: str
    configuration_name: Optional[str]
    model_name: Optional[str]
    dynamic: List[Learning] = field(default_factory=list)
    post_episode: List[Learning] = field(default_factory=list)
    seen_keys: Set[str] = field(default_factory=set)

    def serialize(self, *, success: bool, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "level": self.level_name,
            "configuration": self.configuration_name,
            "model": self.model_name,
            "success": success,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dynamic": [learning.to_dict() for learning in self.dynamic],
            "post_episode": [learning.to_dict() for learning in self.post_episode],
            "stats": stats or {},
        }


class LearningRepository:
    """Simple JSON backed storage for episode learnings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Any] = {"episodes": []}

    def load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except json.JSONDecodeError:
                # Reset corrupt storage but keep file for inspection
                self._data = {"episodes": []}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._data = {"episodes": []}
        if "episodes" not in self._data:
            self._data["episodes"] = []
        return self._data

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add_episode(self, episode: EpisodeContext, *, success: bool, stats: Optional[Dict[str, Any]] = None) -> None:
        data = self.load()
        record = episode.serialize(success=success, stats=stats)
        data.setdefault("episodes", []).append(record)
        self._data = data
        self.persist()

    def get_post_episode_items(
        self,
        *,
        configuration_name: Optional[str] = None,
        level_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        data = self.load()
        episodes: Iterable[Dict[str, Any]] = data.get("episodes", [])
        matched: List[Dict[str, Any]] = []
        for entry in episodes:
            if configuration_name and entry.get("configuration") != configuration_name:
                continue
            if level_name and entry.get("level") != level_name:
                continue
            matched.extend(entry.get("post_episode", []))
        return matched


class LearningManager:
    """Coordinates dynamic and persistent learnings across episodes."""

    def __init__(self, repository: Optional[LearningRepository] = None) -> None:
        storage_path = Path("resources/learnings.json")
        self.repository = repository or LearningRepository(storage_path)
        self.enabled: bool = False
        self._episode: Optional[EpisodeContext] = None
        self._cached_guidelines: List[str] = []
        self._cached_reflections: List[str] = []

    def start_episode(
        self,
        *,
        level_name: str,
        configuration: Optional["Configuration"],
        model: Optional["Model"],
    ) -> None:
        config_name = getattr(configuration, "name", None)
        self.enabled = bool(configuration and getattr(configuration, "learning_feature", False))
        if not self.enabled:
            self._episode = None
            self._cached_guidelines = []
            self._cached_reflections = []
            return

        model_tag: Optional[str] = None
        if model is not None:
            model_spec = getattr(model, "value", None)
            model_tag = getattr(model_spec, "tag", None) or getattr(model_spec, "name", None) or str(model)

        self._episode = EpisodeContext(
            level_name=level_name,
            configuration_name=config_name,
            model_name=model_tag,
        )

        stored_items = self.repository.get_post_episode_items(
            configuration_name=config_name,
            level_name=level_name,
        )
        guidelines: List[str] = []
        reflections: List[str] = []
        for item in stored_items:
            content = item.get("content")
            if not content:
                continue
            category = item.get("metadata", {}).get("category")
            if category == "guideline":
                guidelines.append(content)
            elif category == "reflection":
                reflections.append(content)
            else:
                # default to guideline to ensure it is surfaced
                guidelines.append(content)
        self._cached_guidelines = self._deduplicate(guidelines)
        self._cached_reflections = self._deduplicate(reflections)

    def apply_persistent_learnings(self, memory: "Memory") -> None:
        if not self.enabled or not self._cached_guidelines:
            return
        from llm.memory.memory import Role

        for guideline in self._cached_guidelines:
            memory.add_message(Role.SYSTEM, f"[PERSISTENT META LEARNING] {guideline}")

    def record_dynamic_learning(
        self,
        event: LearningEvent,
        base_message: str,
        *,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not self._episode:
            return

        message = base_message.strip()
        if not message:
            return

        key = f"{event.value}:{message.lower()}"
        if key in self._episode.seen_keys:
            return

        self._episode.seen_keys.add(key)
        dynamic_text = self._format_dynamic_text(event, message, hint)
        metadata = {
            "category": event.value,
            "base_message": message,
        }
        if hint:
            metadata["hint"] = hint
        if context:
            metadata["context"] = context

        self._episode.dynamic.append(
            Learning(
                type=LearningType.DYNAMIC,
                content=dynamic_text,
                metadata=metadata,
            )
        )

    def finalize_episode(
        self,
        *,
        success: Optional[bool],
        run_result: Optional["RunResult"],
    ) -> Optional[str]:
        if not self.enabled or not self._episode:
            return None

        resolved_success = bool(success)
        if success is None and run_result is not None:
            resolved_success = bool(getattr(run_result, "success", 0) > 0)

        summary = self._build_summary(resolved_success, run_result)
        if summary:
            self._episode.post_episode.append(
                Learning(
                    type=LearningType.POST_EPISODE,
                    content=summary,
                    metadata={"category": "reflection"},
                )
            )

        guidelines = self._build_guidelines(resolved_success)
        for guideline in guidelines:
            self._episode.post_episode.append(
                Learning(
                    type=LearningType.POST_EPISODE,
                    content=guideline,
                    metadata={"category": "guideline"},
                )
            )

        stats_dict: Optional[Dict[str, Any]] = None
        if run_result is not None:
            try:
                stats_dict = asdict(run_result)
            except TypeError:
                stats_dict = None

        self.repository.add_episode(
            self._episode,
            success=resolved_success,
            stats=stats_dict,
        )

        # Reset for the next episode
        self._episode = None
        self._cached_guidelines = []
        self._cached_reflections = []

        return summary

    def get_guideline_prompts(self) -> List[str]:
        return list(self._cached_guidelines)

    def get_reflection_notes(self) -> List[str]:
        return list(self._cached_reflections)

    def _format_dynamic_text(
        self,
        event: LearningEvent,
        message: str,
        hint: Optional[str],
    ) -> str:
        cleaned = message.strip()
        if cleaned and cleaned[-1] in ".!?":
            cleaned = cleaned[:-1]

        suffix = ""
        if hint:
            suffix = f" Hint: {hint}"
        if event is LearningEvent.FORMAL_ERROR:
            return (
                "Formal error encountered. Ensure parameter validation and rule compliance before repeating similar "
                f"actions. Detail: {cleaned}.{suffix}"
            )
        if event is LearningEvent.ACTION_FAILURE:
            return (
                "Action failed during execution. Re-evaluate prerequisites or environmental conditions before retrying. "
                f"Detail: {cleaned}.{suffix}"
            )
        if event is LearningEvent.ACTION_SUCCESS:
            return (
                "Successful strategy observed. Consider reusing this approach when context aligns. "
                f"Detail: {cleaned}.{suffix}"
            )
        return f"Observation: {cleaned}.{suffix}"

    def _build_guidelines(self, success: bool) -> List[str]:
        if not self._episode:
            return []
        guidelines: List[str] = []
        for learning in self._episode.dynamic:
            category = learning.metadata.get("category")
            base_message = learning.metadata.get("base_message", learning.content)
            if category == LearningEvent.FORMAL_ERROR.value:
                guidelines.append(
                    "Validate preconditions and tool parameters to avoid formal errors such as: "
                    f"{base_message}."
                )
            elif category == LearningEvent.ACTION_FAILURE.value:
                guidelines.append(
                    "Before committing to an action, confirm the environment allows it to prevent failures like: "
                    f"{base_message}."
                )
            elif category == LearningEvent.ACTION_SUCCESS.value:
                guidelines.append(
                    "Leverage proven tactics. Successful action noted: "
                    f"{base_message}."
                )
            else:
                guidelines.append(base_message)
        if not guidelines:
            outcome = "Succeeded" if success else "Did not succeed"
            guidelines.append(f"Episode outcome: {outcome}. Continue refining strategies.")
        return self._deduplicate(guidelines)

    def _build_summary(self, success: bool, run_result: Optional["RunResult"]) -> Optional[str]:
        if not self._episode:
            return None
        lines: List[str] = [
            "Critical reflection after episode:",
            f"Outcome: {'success' if success else 'failure'}",
        ]
        if run_result is not None:
            lines.append(
                "Performance metrics: "
                f"toolcalls={getattr(run_result, 'toolcall_count', 0)}, "
                f"observations={getattr(run_result, 'observation_count', 0)}, "
                f"soft_errors={getattr(run_result, 'softerror_count', 0)}, "
                f"hard_errors={getattr(run_result, 'harderror_count', 0)}"
            )
        if self._episode.dynamic:
            lines.append("Key learnings observed during the episode:")
            for learning in self._episode.dynamic:
                lines.append(f"- {learning.content}")
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(items: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(normalized)
        return ordered


_LEARNING_MANAGER: Optional[LearningManager] = None


def get_learning_manager() -> LearningManager:
    global _LEARNING_MANAGER
    if _LEARNING_MANAGER is None:
        _LEARNING_MANAGER = LearningManager()
    return _LEARNING_MANAGER
