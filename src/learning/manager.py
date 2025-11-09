"""Review-centric learning manager used by the agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.benchresult import RunResult
    from config import Configuration
    from llm.memory.memory import Memory
    from llm.model import Model


class LearningEvent(str, Enum):
    """Types of telemetry events that can reach the learning manager."""

    FORMAL_ERROR = "formal_error"
    ACTION_FAILURE = "action_failure"
    ACTION_SUCCESS = "action_success"
    REFLECTION_WINDOW = "reflection_window"
    REVIEW = "review"


class LearningType(str, Enum):
    """Supported persistence layers for learnings."""

    DYNAMIC = "dynamic"
    POST_EPISODE = "post_episode"


@dataclass
class Learning:
    """Single learning item archived during an episode."""

    content: str
    category: str
    persistence: LearningType
    source: Optional[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category,
            "persistence": self.persistence.value,
            "source": self.source,
            "created_at": self.created_at,
        }


@dataclass
class ReviewLog:
    """Metadata describing a review prompt emitted during the episode."""

    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    resolved: bool = False
    resolved_at: Optional[str] = None
    learnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
            "learnings": list(self.learnings),
        }


@dataclass
class LearningSuggestion:
    """Structured prompt instructing the agent to perform a review."""

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


class LearningRepository:
    """Simple JSON-backed storage for episodes and ratings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if not self._data:
            if self.path.exists():
                try:
                    self._data = json_load(self.path)
                except ValueError:
                    self._data = {"episodes": [], "ratings": {}}
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._data = {"episodes": [], "ratings": {}}
        if "episodes" not in self._data:
            self._data["episodes"] = []
        if "ratings" not in self._data:
            self._data["ratings"] = {}
        return self._data

    def persist(self) -> None:
        data = self.load()
        json_dump(self.path, data)

    def append_episode(self, record: Dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("episodes", []).append(record)
        self._data = data
        self.persist()

    def get_rating(self, content: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        key = content.strip().lower()
        ratings = data.get("ratings", {})
        return ratings.get(key)

    def update_rating(
        self,
        content: str,
        *,
        label: str,
        score: float,
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = self.load()
        key = content.strip().lower()
        entry = {
            "label": label,
            "rating": float(score),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if rationale:
            entry["rationale"] = rationale
        data.setdefault("ratings", {})[key] = entry
        self._data = data
        self.persist()
        return entry

    def get_guidelines(
        self,
        *,
        configuration_name: Optional[str],
        level_name: Optional[str],
    ) -> List[Dict[str, Any]]:
        data = self.load()
        episodes: Iterable[Dict[str, Any]] = data.get("episodes", [])
        seen: Set[str] = set()
        guidelines: List[Dict[str, Any]] = []
        for episode in reversed(list(episodes)):
            if configuration_name and episode.get("configuration") != configuration_name:
                continue
            if level_name and episode.get("level") != level_name:
                continue
            payload = episode.get("learnings", {})
            for item in payload.get(LearningType.POST_EPISODE.value, []):
                if item.get("category") != "guideline":
                    continue
                content = item.get("content", "").strip()
                if not content:
                    continue
                key = content.lower()
                if key in seen:
                    continue
                seen.add(key)
                rating_info = data.get("ratings", {}).get(key)
                guidelines.append(
                    {
                        "content": content,
                        "label": rating_info.get("label") if rating_info else None,
                        "rating": rating_info.get("rating") if rating_info else None,
                    }
                )
        guidelines.sort(
            key=lambda entry: (
                -float(entry.get("rating") or 0.0),
                entry.get("content", ""),
            )
        )
        return guidelines


def json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def json_dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


class LearningManager:
    """Coordinates review prompts and persistent learnings."""

    _RATING_SCALE: Dict[str, float] = {
        "core": 3.0,
        "useful": 2.0,
        "niche": 1.0,
        "retire": 0.0,
    }

    def __init__(self, repository: Optional[LearningRepository] = None) -> None:
        storage_path = Path("learnings.json")
        self.repository = repository or LearningRepository(storage_path)
        self.enabled: bool = False
        self._episode_context: Optional[Dict[str, Optional[str]]] = None
        self._events_since_review: List[Dict[str, Any]] = []
        self._suggestions: Dict[str, LearningSuggestion] = {}
        self._used_suggestions: Set[str] = set()
        self._review_counter = count(1)
        self._episode_learnings: Dict[LearningType, List[Learning]] = {
            LearningType.DYNAMIC: [],
            LearningType.POST_EPISODE: [],
        }
        self._episode_reviews: Dict[str, ReviewLog] = {}
        self._cached_guidelines: List[Dict[str, Any]] = []
        self._review_window: int = 3

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def start_episode(
        self,
        *,
        level_name: str,
        configuration: Optional["Configuration"],
        model: Optional["Model"],
    ) -> None:
        config_name = getattr(configuration, "name", None)
        self.enabled = bool(configuration and getattr(configuration, "learning_feature", False))
        self._episode_context = {
            "level": level_name,
            "configuration": config_name,
            "model": getattr(getattr(model, "value", None), "tag", None) if model else None,
        }

        self._events_since_review = []
        self._suggestions = {}
        self._used_suggestions = set()
        self._review_counter = count(1)
        self._episode_learnings = {
            LearningType.DYNAMIC: [],
            LearningType.POST_EPISODE: [],
        }
        self._episode_reviews = {}
        self._cached_guidelines = []

        if not self.enabled:
            return

        self._cached_guidelines = self.repository.get_guidelines(
            configuration_name=config_name,
            level_name=level_name,
        )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------
    def apply_persistent_learnings(self, memory: "Memory") -> None:
        if not self.enabled or not self._cached_guidelines:
            return

        from llm.memory.memory import Role

        entries: List[tuple[Role, str]] = []
        seen: Set[str] = set()
        for item in self._cached_guidelines:
            content = item.get("content", "").strip()
            if not content:
                continue
            label = item.get("label")
            if label:
                message = f"[REVIEW LEARNING::{label.upper()}] {content}"
            else:
                message = f"[REVIEW LEARNING] {content}"
            if message in seen:
                continue
            seen.add(message)
            entries.append((Role.SYSTEM, message))

        if entries:
            memory.extend_at_top(entries)

    # ------------------------------------------------------------------
    # Suggestion handling
    # ------------------------------------------------------------------
    def record_dynamic_learning(
        self,
        event: LearningEvent,
        base_message: str,
        *,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[LearningSuggestion]:
        return self.register_event(event, base_message, hint=hint, context=context)

    def register_event(
        self,
        event: LearningEvent,
        base_message: str,
        *,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[LearningSuggestion]:
        if not self.enabled or not self._episode_context:
            return None

        message = base_message.strip()
        clean_hint = hint.strip() if hint else None
        payload = {
            "event": event.value,
            "message": message,
            "hint": clean_hint,
            "context": self._safe_context(context),
        }

        if event is LearningEvent.REVIEW:
            events = list(self._events_since_review)
            events.append(payload)
            self._events_since_review = []
            return self._emit_review_prompt(events=events, reason="manual_review", note=message or clean_hint)

        self._events_since_review.append(payload)
        if len(self._events_since_review) < self._review_window:
            return None

        events = list(self._events_since_review)
        self._events_since_review = []
        return self._emit_review_prompt(events=events, reason="automatic_window", note=None)

    def get_pending_suggestions(self) -> List[LearningSuggestion]:
        if not self.enabled:
            return []
        return [
            suggestion
            for suggestion_id, suggestion in self._suggestions.items()
            if suggestion_id not in self._used_suggestions
        ]

    # ------------------------------------------------------------------
    # Learning persistence
    # ------------------------------------------------------------------
    def save_learning(
        self,
        *,
        content: str,
        category: str,
        persistence: str,
        source: Optional[str],
    ) -> str:
        if not self.enabled or not self._episode_context:
            return "Learning manager disabled; nothing stored."

        normalized = content.strip()
        if not normalized:
            return "Cannot store an empty learning."

        try:
            persistence_type = LearningType(persistence.lower())
        except ValueError:
            return "Unknown persistence. Use 'dynamic' or 'post_episode'."

        created_at = datetime.utcnow().isoformat() + "Z"
        record = Learning(
            content=normalized,
            category=category or "guideline",
            persistence=persistence_type,
            source=source,
            created_at=created_at,
        )
        self._episode_learnings[persistence_type].append(record)

        if persistence_type is LearningType.POST_EPISODE:
            self._cached_guidelines.insert(
                0,
                {
                    "content": normalized,
                    "label": None,
                    "rating": None,
                },
            )

        if source and source in self._episode_reviews:
            self._used_suggestions.add(source)
            review = self._episode_reviews[source]
            review.resolved = True
            review.resolved_at = created_at
            review.learnings.append(normalized)

        return (
            f"Stored {persistence_type.value} learning. Remember to rate it with `rate_meta_learning`."
        )

    def rate_learning(
        self,
        content: str,
        *,
        label: str,
        score: Optional[float] = None,
        rationale: Optional[str] = None,
    ) -> str:
        if not self.enabled:
            return "Learning manager disabled; rating ignored."

        normalized_label = label.strip().lower()
        if normalized_label not in self._RATING_SCALE and score is None:
            return "Unknown label. Valid labels: core, useful, niche, retire."

        rating_score = float(score) if score is not None else self._RATING_SCALE[normalized_label]
        entry = self.repository.update_rating(
            content,
            label=normalized_label,
            score=rating_score,
            rationale=rationale,
        )

        key = content.strip().lower()
        for guideline in self._cached_guidelines:
            if guideline.get("content", "").strip().lower() == key:
                guideline["label"] = entry.get("label")
                guideline["rating"] = entry.get("rating")
                break

        return f"Learning rated as {entry['label']} ({entry['rating']})."

    # ------------------------------------------------------------------
    # Episode finalisation
    # ------------------------------------------------------------------
    def finalize_episode(
        self,
        *,
        success: Optional[bool],
        run_result: Optional["RunResult"],
    ) -> Optional[str]:
        if not self.enabled or not self._episode_context:
            self._episode_context = None
            return None

        resolved_success = bool(success)
        record = {
            "level": self._episode_context.get("level"),
            "configuration": self._episode_context.get("configuration"),
            "model": self._episode_context.get("model"),
            "success": resolved_success,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "learnings": {
                LearningType.DYNAMIC.value: [item.to_dict() for item in self._episode_learnings[LearningType.DYNAMIC]],
                LearningType.POST_EPISODE.value: [
                    item.to_dict() for item in self._episode_learnings[LearningType.POST_EPISODE]
                ],
            },
            "reviews": [log.to_dict() for log in self._episode_reviews.values()],
        }
        if run_result is not None:
            try:
                record["stats"] = asdict(run_result)
            except TypeError:
                record["stats"] = None

        self.repository.append_episode(record)

        summary_lines = ["Review-driven learning summary:"]
        outcome_text = "Success" if resolved_success else "Objective incomplete"
        summary_lines.append(f"Outcome: {outcome_text}.")

        persistent = self._episode_learnings[LearningType.POST_EPISODE]
        dynamic = self._episode_learnings[LearningType.DYNAMIC]

        if persistent:
            summary_lines.append("Persistent learnings archived:")
            summary_lines.extend(f"- {item.content}" for item in persistent)
        elif dynamic:
            summary_lines.append("Dynamic learnings captured:")
            summary_lines.extend(f"- {item.content}" for item in dynamic)
        else:
            summary_lines.append("No new learnings captured this episode.")

        pending = [log.id for log in self._episode_reviews.values() if not log.resolved]
        if pending:
            summary_lines.append("Pending reviews: " + ", ".join(sorted(pending)))

        self._episode_context = None
        self._events_since_review = []
        self._suggestions = {}
        self._used_suggestions = set()
        self._episode_learnings = {
            LearningType.DYNAMIC: [],
            LearningType.POST_EPISODE: [],
        }
        self._episode_reviews = {}
        self._cached_guidelines = []

        return "\n".join(summary_lines)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_guideline_prompts(self) -> List[str]:
        return [item.get("content", "") for item in self._cached_guidelines]

    def get_reflection_notes(self) -> List[str]:
        return [log.content for log in self._episode_reviews.values()]

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _emit_review_prompt(
        self,
        *,
        events: Sequence[Dict[str, Any]],
        reason: str,
        note: Optional[str],
    ) -> Optional[LearningSuggestion]:
        if not events:
            return None

        review_id = f"R{next(self._review_counter)}"
        created_at = datetime.utcnow().isoformat() + "Z"
        rating_text = ", ".join(
            f"{label}={int(value) if float(value).is_integer() else value}"
            for label, value in self._RATING_SCALE.items()
        )

        lines = [
            f"Review request {review_id}: consolidate transferable insights from the recent gameplay.",
            "Steps:",
            "1. Call `list_learning_suggestions` to reread queued reviews as needed.",
            "2. Discuss possible lessons in natural language and annotate candidates as `LEARNING:` lines.",
            f"3. Persist confident guidance with `store_meta_learning` (persistence='post_episode', source='{review_id}').",
            "4. Immediately rate each stored learning via `rate_meta_learning` (labels: core/useful/niche/retire).",
            f"Rating scale reminder: {rating_text}.",
        ]
        if note:
            lines.append(f"Reviewer note: {note}")
        lines.append("Recent context:")
        for idx, entry in enumerate(events, start=1):
            details = entry.get("message") or "(no message)"
            hint = entry.get("hint")
            context = entry.get("context")
            line = f"{idx}. {entry.get('event')} – {details}"
            if hint:
                line += f" | hint: {hint}"
            if context:
                line += f" | context: {context}"
            lines.append(f"   {line}")

        if self._cached_guidelines:
            lines.append("Existing high-signal guidelines:")
            for item in self._cached_guidelines[:3]:
                label = item.get("label")
                rating = item.get("rating")
                suffix = ""
                if label and rating is not None:
                    suffix = f" [{label}:{rating}]"
                elif label:
                    suffix = f" [{label}]"
                elif rating is not None:
                    suffix = f" [score={rating}]"
                lines.append(f"- {item.get('content', '')}{suffix}")

        suggestion = LearningSuggestion(
            id=review_id,
            content="\n".join(lines),
            event=LearningEvent.REVIEW,
            metadata={
                "created_at": created_at,
                "events": events,
                "reason": reason,
                **({"note": note} if note else {}),
            },
        )

        self._suggestions[review_id] = suggestion
        self._episode_reviews[review_id] = ReviewLog(
            id=review_id,
            content=suggestion.content,
            metadata=suggestion.metadata,
            created_at=created_at,
        )
        return suggestion

    @staticmethod
    def _safe_context(context: Optional[Dict[str, Any]], depth: int = 2) -> Any:
        if depth <= 0:
            return None
        if context is None:
            return {}
        if isinstance(context, dict):
            return {str(key): LearningManager._safe_context(value, depth - 1) for key, value in context.items()}
        if isinstance(context, list):
            return [LearningManager._safe_context(item, depth - 1) for item in context]
        if isinstance(context, (str, int, float, bool)) or context is None:
            return context
        return str(context)


_LEARNING_MANAGER: Optional[LearningManager] = None


def get_learning_manager() -> LearningManager:
    global _LEARNING_MANAGER
    if _LEARNING_MANAGER is None:
        _LEARNING_MANAGER = LearningManager()
    return _LEARNING_MANAGER
