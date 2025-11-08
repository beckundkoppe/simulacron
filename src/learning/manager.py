"""Learning management for dynamic and post-episode meta learnings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from itertools import count
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
    suggestion_index: Dict[str, str] = field(default_factory=dict)
    suggestions: Dict[str, "LearningSuggestion"] = field(default_factory=dict)
    used_suggestions: Set[str] = field(default_factory=set)

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
            "reflection_prompts": [suggestion.to_dict() for suggestion in self.suggestions.values()],
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
        storage_path = Path("learnings.json")
        self.repository = repository or LearningRepository(storage_path)
        self.enabled: bool = False
        self._episode: Optional[EpisodeContext] = None
        self._cached_guidelines: List[str] = []
        self._cached_reflections: List[str] = []
        self._suggestion_counter = count(1)

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
        self._suggestion_counter = count(1)

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
    ) -> Optional["LearningSuggestion"]:
        """Backward compatible entry point that now routes to reflection prompts."""

        return self.register_event(event, base_message, hint=hint, context=context)

    def register_event(
        self,
        event: LearningEvent,
        base_message: str,
        *,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional["LearningSuggestion"]:
        if not self.enabled or not self._episode:
            return None

        suggestion = self._build_reflection_prompt(event, base_message, hint=hint, context=context)
        if suggestion is None:
            return None

        if suggestion.id in self._episode.suggestions:
            return self._episode.suggestions[suggestion.id]

        key = suggestion.metadata.get("dedupe_key")
        if key:
            if key in self._episode.suggestion_index:
                existing_id = self._episode.suggestion_index[key]
                return self._episode.suggestions.get(existing_id)
            self._episode.suggestion_index[key] = suggestion.id

        self._episode.suggestions[suggestion.id] = suggestion
        return suggestion

    def get_pending_suggestions(self) -> List["LearningSuggestion"]:
        if not self.enabled or not self._episode:
            return []

        pending: List[LearningSuggestion] = []
        for suggestion_id, suggestion in self._episode.suggestions.items():
            if suggestion_id in self._episode.used_suggestions:
                continue
            pending.append(suggestion)
        return pending

    def save_learning(
        self,
        *,
        content: str,
        category: str = "guideline",
        persistence: str = "dynamic",
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.enabled or not self._episode:
            return "Learning feature is disabled for this configuration."

        normalized_content = content.strip()
        if not normalized_content:
            return "Learning content must not be empty."

        try:
            learning_type = LearningType(persistence)
        except ValueError:
            return (
                "Unknown persistence value. Use 'dynamic' for immediate application or "
                "'post_episode' for after-action learnings."
            )

        target_collection: List[Learning] = getattr(self._episode, learning_type.value)

        key = f"learning:{normalized_content.lower()}"
        if key in self._episode.seen_keys:
            return "This learning has already been recorded."

        metadata: Dict[str, Any] = {
            "category": category,
            "base_message": normalized_content,
        }
        if context:
            metadata["context"] = context

        if source:
            if source in self._episode.suggestions:
                self._episode.used_suggestions.add(source)
            metadata["prompt_source"] = source

        target_collection.append(
            Learning(
                type=learning_type,
                content=normalized_content,
                metadata=metadata,
            )
        )
        self._episode.seen_keys.add(key)

        return (
            "Learning stored successfully. It will be reused during this episode "
            "and future runs depending on its persistence."
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
    def _build_reflection_prompt(
        self,
        event: LearningEvent,
        base_message: str,
        *,
        hint: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Optional["LearningSuggestion"]:
        message = base_message.strip()
        if not message:
            return None

        prompt_lines: List[str] = []
        if event is LearningEvent.FORMAL_ERROR:
            prompt_lines.append(
                "A formal error occurred. Analyse why the action violated the rules or parameters and describe the fix."
            )
        elif event is LearningEvent.ACTION_FAILURE:
            prompt_lines.append(
                "An attempted action failed. Diagnose missing prerequisites or misunderstandings before retrying."
            )
        elif event is LearningEvent.ACTION_SUCCESS:
            prompt_lines.append(
                "A successful action was observed. Extract the reusable tactic so it can be repeated intentionally."
            )
        else:
            prompt_lines.append("Review the recent event and capture any reusable guidance.")

        prompt_lines.append(f"Event detail: {message}")
        if hint:
            prompt_lines.append(f"Hint: {hint}")
        prompt_lines.append(
            "Respond by articulating the insight and, when confident, persist it using the `store_meta_learning` tool."
        )

        metadata: Dict[str, Any] = {
            "event": event.value,
            "base_message": message,
            "dedupe_key": self._make_suggestion_key(event, message, hint=hint),
        }
        if hint:
            metadata["hint"] = hint
        if context:
            metadata["context"] = context

        suggestion = LearningSuggestion(
            id=f"S{next(self._suggestion_counter)}",
            content=" ".join(prompt_lines),
            event=event,
            metadata=metadata,
        )
        return suggestion

    @staticmethod
    def _make_suggestion_key(
        event: LearningEvent,
        message: str,
        *,
        hint: Optional[str] = None,
    ) -> str:
        key_parts = [event.value, message.lower().strip()]
        if hint:
            key_parts.append(f"hint:{hint.lower().strip()}")
        return "|".join(key_parts)


@dataclass
class LearningSuggestion:
    """Reflection prompt that nudges the agent to derive a meta learning."""

    id: str
    content: str
    event: LearningEvent
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "event": self.event.value,
            "metadata": self.metadata,
        }


_LEARNING_MANAGER: Optional[LearningManager] = None


def get_learning_manager() -> LearningManager:
    global _LEARNING_MANAGER
    if _LEARNING_MANAGER is None:
        _LEARNING_MANAGER = LearningManager()
    return _LEARNING_MANAGER
