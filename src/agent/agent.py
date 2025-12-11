import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from agent.helper import process_action_results, process_formal_errors
from agent.plan import FreePlan, PlanNode, StepPlan, TreePlan
from agent.toolpool import ToolGroup, register_tools
import config
import current
import debug
from enviroment.entity import Entity
from enviroment.resultbuffer import FormalError, Resultbuffer, ActionNotPossible, Success
from llm.memory.memory import Memory, Role, Type
from llm.memory.supermem import SuperMemory
from llm.model import Model
from llm.provider import Provider
from llm.toolprovider import ToolProvider
from util.console import Color, banner, bullet, pretty

@dataclass
class BeliefState:
    goals: List[str]
    facts: Dict[str, Any]
    uncertainties: List[str]
    history_summary: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Commitment:
    subgoal: str
    action_type: str
    success_criteria: str
    max_exec_steps: int
    reason: str = ""

@dataclass
class ActionCandidate:
    call: str
    reason: str = ""

_BDI_POLICY_MESSAGE = (
    "You are the execution policy of an agent in a dynamic graph.\n"
    "You get a local graph view, a sub-goal and an execution context.\n\n"
    "Tasks:\n"
    "- Choose exactly one next action from the allowed primitive actions.\n"
    "- Update the exec context (path, notes, status).\n"
    "- Decide whether execution is done (exec_done).\n"
)

class Agent:
    def __init__(self, goal: str, entity: Entity, imaginator_model: Model, realisator_model: Model):
        self.entity = entity
        self.imaginator_model = imaginator_model
        self.realisator_model = realisator_model

        self.goal = goal
        self.plan = None
        self.main_memory = self._create_main_memory(goal)

        self.main_memory.append_message(
            Role.SYSTEM,
                "You control a human in a 2D room-based environment. The human agent can move, take and drop objects, and interact with objects (open, close, lock, unlock, go through or look through doors). Always choose the minimal action sequence that achieves the requsted goal and nothing else.",
            Type.GOAL
        )

        self.use_bdi = config.ACTIVE_CONFIG.agent.action in (
            config.ActionType.BDI,
            config.ActionType.BDI_EXPLORATION,
        )
        self.use_ranked_actions = config.ACTIVE_CONFIG.agent.action is config.ActionType.RANKED_ACTIONS
        self.use_react = config.ACTIVE_CONFIG.agent.action is config.ActionType.REACT
        self.exploration_mode = False
        self.belief_state: Optional[BeliefState] = None
        self.current_commitment: Optional[Commitment] = None
        self.bdi_last_feedback: str = ""

        if self.use_ranked_actions:
            self.main_memory.append_message(
                Role.SYSTEM,
                "Ranked action loop active: propose multiple primitive tool calls with short justifications, pick the best, then execute only that choice.",
                Type.GOAL,
            )
        if self.use_react:
            self.main_memory.append_message(
                Role.SYSTEM,
                "ReAct loop active: think step-by-step using the latest perception and feedback, then pick exactly one primitive tool call to execute.",
                Type.GOAL,
            )

        self.had_action = False
        self.triggered_replan = None
        self.finished = False

        if self.use_bdi:
            self._init_bdi_state()

        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            self.triggered_replan = "Main goal is not yet reached. Create an initial plan"

        debug.print_to_file(self.main_memory.get_history())

    def _add_time(self, field: str, delta: float) -> None:
        result = getattr(current, "RESULT", None)
        if result is None:
            return
        setattr(result, field, getattr(result, field) + delta)

    def _create_main_memory(self, goal: str) -> Memory:
        mem_type = getattr(config.ACTIVE_CONFIG.agent, "memory_type", None)

        if mem_type is config.MemoryType.SUPER:
            return SuperMemory(goal=goal, path="main_memory.txt")

        return Memory(goal=goal, path="main_memory.txt")

    def _init_bdi_state(self) -> None:
        self.belief_state = BeliefState(
            goals=[self.goal],
            facts={},
            uncertainties=[],
            history_summary="",
            meta={"exec_notes": [], "exec_path": [], "status": "init"},
        )
        self.current_commitment = None
        self.exploration_mode = False
        self.main_memory.append_message(Role.SYSTEM, _BDI_POLICY_MESSAGE, Type.GOAL)
        self.main_memory.append_message(
            Role.SYSTEM,
            "BDI loop active: keep answers short, use only allowed tool calls, and update execution status concisely.",
            Type.GOAL,
        )

    def _update_bdi_loop(self, perception: str) -> None:
        if not self.belief_state:
            self._init_bdi_state()

        self.main_memory.append_message(Role.USER, perception, Type.PERCEPTION)
        feedback = self._bdi_collect_feedback()

        t0 = time.time()
        self._bdi_brf(perception, feedback)
        self._add_time("observe_time_s", time.time() - t0)

        t0 = time.time()
        options = self._bdi_generate_options()
        self._add_time("plan_time_s", time.time() - t0)

        t0 = time.time()
        commitment = self._bdi_filter_intention(options)
        self._add_time("trial_time_s", time.time() - t0)

        if commitment is None:
            self.main_memory.append_message(
                Role.USER,
                "No actionable intention derived from current belief state; waiting for better context.",
                Type.FEEDBACK,
            )
            self.main_memory.save()
            return

        self.current_commitment = commitment
        self.main_memory.append_message(
            Role.USER,
            f"[Intention] {commitment.subgoal} | success: {commitment.success_criteria} | budget: {commitment.max_exec_steps}",
            Type.PLAN,
        )

        t0 = time.time()
        self._bdi_execute_commitment(commitment, perception)
        self._add_time("action_time_s", time.time() - t0)
        self.had_action = True
        self.main_memory.save()

    def _bdi_collect_feedback(self) -> str:
        """Ingest recent tool feedback into memory and return it for belief updates."""
        process_formal_errors(self.main_memory)
        collected: list[str] = []
        for role, msg, msg_type in process_action_results():
            collected.append(msg)
            self.main_memory.append_message(role, msg, msg_type)

        feedback = "\n".join(collected).strip()
        self.bdi_last_feedback = feedback
        return feedback

    def _safe_json_parse(self, raw: str) -> Optional[Any]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
        return None

    def _perception_ids(self, perception: str) -> list[str]:
        """Extract known readable_ids from the current perception JSON."""
        try:
            data = json.loads(perception)
        except Exception:
            return []

        ids: list[str] = []
        for item in data.get("your_perception", []):
            rid = item.get("id")
            if rid:
                ids.append(str(rid))
            contents = item.get("contents")
            if isinstance(contents, list):
                for c in contents:
                    if isinstance(c, dict):
                        rid2 = c.get("id")
                        if rid2:
                            ids.append(str(rid2))
        for inv in data.get("your_inventory", []):
            if isinstance(inv, dict):
                iid = inv.get("id")
                if iid:
                    ids.append(str(iid))
        return ids

    def _perception_lookup(self, perception: str) -> tuple[dict[str, str], list[str]]:
        """Extract id->name mapping and inventory ids from a perception blob."""
        names: dict[str, str] = {}
        inventory_ids: list[str] = []

        try:
            data = json.loads(perception)
        except Exception:
            return names, inventory_ids

        for item in data.get("your_perception", []):
            rid = item.get("id")
            name = item.get("name")
            if rid and name:
                names[str(rid)] = str(name)
            contents = item.get("contents")
            if isinstance(contents, list):
                for c in contents:
                    if isinstance(c, dict):
                        cid = c.get("id")
                        cname = c.get("name")
                        if cid and cname:
                            names[str(cid)] = str(cname)

        for inv in data.get("your_inventory", []):
            if isinstance(inv, dict):
                iid = inv.get("id")
                iname = inv.get("name")
                if iid:
                    inventory_ids.append(str(iid))
                if iid and iname and str(iid) not in names:
                    names[str(iid)] = str(iname)

        return names, inventory_ids

    def _parse_positions(self, perception: str) -> tuple[Optional[tuple[float, float]], dict[str, tuple[float, float]]]:
        """Parse agent and object positions from perception JSON."""
        agent_pos: Optional[tuple[float, float]] = None
        pos_map: dict[str, tuple[float, float]] = {}

        try:
            data = json.loads(perception)
        except Exception:
            return agent_pos, pos_map

        you = data.get("you_are_in_room", {})
        your_pos = you.get("your_pos") or {}
        try:
            agent_pos = (float(your_pos.get("x")), float(your_pos.get("y")))
        except Exception:
            agent_pos = None

        def parse_point(val: Any) -> Optional[tuple[float, float]]:
            if isinstance(val, dict):
                try:
                    return (float(val.get("x")), float(val.get("y")))
                except Exception:
                    return None
            if isinstance(val, str):
                try:
                    clean = val.strip("()")
                    x_str, y_str = clean.split(",")
                    return (float(x_str), float(y_str))
                except Exception:
                    return None
            return None

        for item in data.get("your_perception", []):
            rid = item.get("id")
            pos = parse_point(item.get("position"))
            if rid and pos:
                pos_map[str(rid)] = pos

        return agent_pos, pos_map

    def _parse_tool_calls(self, text: str) -> tuple[list[str], bool]:
        """Parse planned tool calls into a list; return (calls, is_json)."""
        text = text.strip()
        if not text:
            return [], False

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                calls = [str(x).strip() for x in parsed if str(x).strip()]
                return calls, True
        except Exception:
            pass

        lines = [ln.strip(" \t,") for ln in text.splitlines() if ln.strip(" \t,")]
        if len(lines) == 1 and ";" in lines[0]:
            lines = [seg.strip() for seg in lines[0].split(";") if seg.strip()]
        return lines, False

    def _serialize_tool_calls(self, calls: list[str], prefer_json: bool) -> str:
        if not calls:
            return ""
        if prefer_json:
            return json.dumps(calls, ensure_ascii=True)
        return "\n".join(calls)

    def _looks_like_key(self, obj_id: str, names: dict[str, str]) -> bool:
        low_id = obj_id.lower()
        if "key" in low_id:
            return True
        name = names.get(obj_id, "")
        return "key" in name.lower()

    def _deterministic_action_critic(
        self,
        calls: list[str],
        allowed_ids: list[str],
        inventory_ids: list[str],
        names: dict[str, str],
        agent_pos: Optional[tuple[float, float]] = None,
        pos_by_id: Optional[dict[str, tuple[float, float]]] = None,
        goal_ids: Optional[set[str]] = None,
    ) -> list[str]:
        """Deterministic guardrails: normalize calls, drop unknown/hallucinated, and enforce ids/keys."""
        allowed = set(str(x) for x in allowed_ids) | set(inventory_ids)
        goal_ids = goal_ids or set()
        inventory_keys = [iid for iid in inventory_ids if self._looks_like_key(iid, names)]

        filtered: list[str] = []
        id_pattern = re.compile(r"([A-Za-z]+_\d+)")

        def normalize(call: str) -> Optional[str]:
            raw = call.strip().replace(",", " ").replace(";", " ")
            if not raw:
                return None
            tokens = [t for t in raw.split() if t]
            if not tokens:
                return None

            verb = tokens[0].lower()
            rest = tokens[1:]

            def first_id(candidates: list[str]) -> Optional[str]:
                for t in candidates:
                    if id_pattern.fullmatch(t):
                        return t
                return None

            if verb in {"move_to_object", "move_to", "move", "go", "walk", "goto"}:
                rest_lower = [r.lower() for r in rest]
                rest_ids = [t for t in rest if id_pattern.fullmatch(t)]
                if "through" in rest_lower and any("door" in rid for rid in rest_ids):
                    target = next((rid for rid in rest_ids if "door" in rid), None)
                    if target:
                        return f"interact_with_object {target} GO_THROUGH_DOOR"
                target = first_id(rest)
                if not target:
                    return None
                return f"move_to_object {target}"

            if verb in {"go_through", "go_through_door", "through"}:
                target = first_id(rest)
                if not target or "door" not in target:
                    return None
                return f"interact_with_object {target} GO_THROUGH_DOOR"

            if verb in {"open", "close"}:
                target = first_id(rest)
                if not target:
                    return None
                return f"interact_with_object {target} {verb.upper()}"

            if verb in {"unlock", "lock"}:
                target = first_id(rest)
                key = None
                # Prefer explicit key tokens in the call.
                for t in rest:
                    if self._looks_like_key(t, names):
                        key = t
                        break
                if not target or (verb in {"unlock", "lock"} and not key and not inventory_keys):
                    return None
                if not key and inventory_keys:
                    key = inventory_keys[0]
                return f"interact_with_object_using_item {target} {key} {verb.upper()}"

            if verb in {"take", "pickup", "pick", "grab"}:
                what = first_id(rest)
                if not what:
                    return None
                # Look for a source after "from"
                from_id = None
                if "from" in [r.lower() for r in rest]:
                    try:
                        idx = [r.lower() for r in rest].index("from")
                        from_id = first_id(rest[idx + 1 :])
                    except Exception:
                        from_id = None
                if not from_id:
                    from_id = "FLOOR"
                return f"take_from {what} {from_id}"

            if verb in {"drop", "place", "put"}:
                what = first_id(rest)
                if not what:
                    return None
                to_id = None
                if "into" in [r.lower() for r in rest] or "onto" in [r.lower() for r in rest] or "on" in [r.lower() for r in rest]:
                    lowered = [r.lower() for r in rest]
                    for marker in ("into", "onto", "on"):
                        if marker in lowered:
                            try:
                                idx = lowered.index(marker)
                                to_id = first_id(rest[idx + 1 :])
                                break
                            except Exception:
                                to_id = None
                if not to_id:
                    to_id = "FLOOR"
                return f"drop_to {what} {to_id}"

            if verb in {"interact_with_object_using_item", "interact_with_object"}:
                return raw

            return None

        for call in calls:
            normalized = normalize(call)
            if not normalized:
                continue

            ids_in_call = id_pattern.findall(normalized)
            if ids_in_call and any(i not in allowed and i not in goal_ids for i in ids_in_call):
                continue

            lowered = normalized.lower()
            if "unlock" in lowered or "lock" in lowered:
                has_key = any(self._looks_like_key(i, names) for i in ids_in_call)
                if not has_key:
                    if inventory_keys:
                        normalized = f"{normalized} using {inventory_keys[0]}"
                        ids_in_call.append(inventory_keys[0])
                        allowed.add(inventory_keys[0])
                    else:
                        continue

            if normalized.startswith("move_to_object"):
                parts = normalized.split()
                target = parts[1] if len(parts) > 1 else None
                if agent_pos and pos_by_id and target in pos_by_id:
                    tx, ty = pos_by_id[target]
                    ax, ay = agent_pos
                    dist = math.dist((ax, ay), (tx, ty))
                    if dist <= getattr(config.ACTIVE_CONFIG, "interaction_distance", 1.5) + 1e-6:
                        continue

            if filtered and normalized == filtered[-1]:
                continue

            filtered.append(normalized)

        return filtered

    def _apply_action_critic(
        self,
        imagination: str,
        commitment: Commitment,
        perception: str,
        allowed_ids: list[str],
    ) -> str:
        """BDI-only action critic that validates planned tool calls before execution."""
        names, inventory_ids = self._perception_lookup(perception)
        allowed_set = sorted(set(allowed_ids or []))
        allowed_serialized = ", ".join(allowed_set) if allowed_set else "none-visible"
        original_calls, original_json = self._parse_tool_calls(imagination)
        agent_pos, pos_by_id = self._parse_positions(perception)
        goal_ids = set(re.findall(r"[A-Za-z]+_\\d+", f"{commitment.subgoal} {commitment.success_criteria} {commitment.reason}"))
        deterministically_filtered = self._deterministic_action_critic(
            original_calls,
            allowed_ids=allowed_set,
            inventory_ids=inventory_ids,
            names=names,
            agent_pos=agent_pos,
            pos_by_id=pos_by_id,
            goal_ids=goal_ids,
        )
        filtered_imagination = self._serialize_tool_calls(deterministically_filtered, prefer_json=original_json)

        prompt = (
            "You are an action-level critic. Validate/repair the planned primitive tool calls before execution.\n"
            f"Commitment: {commitment.subgoal} | success: {commitment.success_criteria}\n"
            f"Allowed IDs: {allowed_serialized}\n"
            f"Inventory IDs: {', '.join(inventory_ids) or 'empty'}\n"
            f"Known object names: {json.dumps(names, ensure_ascii=True)}\n"
            f"Last feedback: {self.bdi_last_feedback or 'none'}\n"
            f"Planned tool calls:\n{filtered_imagination or imagination}\n"
            "Rules:\n"
            "- Use only IDs from Allowed IDs.\n"
            "- For lock/unlock actions, only use items that look like keys (id or name contains 'key'). "
            "If the plan uses a non-key item, replace it with the best available key; if none exists, drop that lock/unlock step.\n"
            "- Keep steps minimal, preserve valid ordering, and return ONLY the corrected tool calls. "
            "If everything is valid, repeat the original list verbatim."
        )

        critic = Provider.build("bdi_action_critic", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = critic.call(prompt)
        self._add_time("img_time_s", time.time() - start)

        llm_calls, llm_json = self._parse_tool_calls(reply)
        llm_filtered = self._deterministic_action_critic(
            llm_calls,
            allowed_ids=allowed_set,
            inventory_ids=inventory_ids,
            names=names,
            agent_pos=agent_pos,
            pos_by_id=pos_by_id,
            goal_ids=goal_ids,
        )

        final_calls = llm_filtered or deterministically_filtered
        final_json = llm_json if llm_filtered else original_json
        if not final_calls:
            return ""
        return self._serialize_tool_calls(final_calls, prefer_json=final_json)

    def _format_belief_state(self, compact: bool = False) -> str:
        if not self.belief_state:
            return "no belief state"

        facts = "; ".join([f"{k}: {v}" for k, v in self.belief_state.facts.items()]) or "none"
        uncertainties = "; ".join(self.belief_state.uncertainties) or "none"
        meta = "; ".join([f"{k}: {v}" for k, v in self.belief_state.meta.items()]) or "none"
        goals = "; ".join(self.belief_state.goals)

        if compact:
            return (
                f"goals={goals} | facts={facts} | uncertainties={uncertainties} | "
                f"summary={self.belief_state.history_summary} | meta={meta} | exploration={self.exploration_mode}"
            )

        return (
            f"Goals: {goals}\n"
            f"Facts: {facts}\n"
            f"Uncertainties: {uncertainties}\n"
            f"History: {self.belief_state.history_summary}\n"
            f"Meta: {meta}\n"
            f"Exploration: {self.exploration_mode}"
        )

    def _commitment_from_dict(self, data: Dict[str, Any]) -> Optional[Commitment]:
        try:
            subgoal = str(data.get("subgoal") or data.get("goal") or data.get("task") or "").strip()
            if not subgoal:
                return None

            success = str(data.get("success_criteria") or data.get("done") or "goal advanced").strip()
            action_type = str(data.get("action_type") or data.get("action") or "env_step").strip()
            max_steps_raw = data.get("max_exec_steps") or data.get("max_steps") or 1
            max_steps = int(max(1, min(5, int(max_steps_raw))))
            reason = str(data.get("reason") or data.get("why") or "").strip()

            return Commitment(
                subgoal=subgoal,
                action_type=action_type,
                success_criteria=success,
                max_exec_steps=max_steps,
                reason=reason,
            )
        except Exception:
            return None

    def _bdi_brf(self, perception: str, feedback: str) -> None:
        if not self.belief_state:
            return

        prompt = (
            f"Current belief state:\n{self._format_belief_state()}\n\n"
            f"New perception:\n{perception}\n\n"
            f"Last feedback: {feedback or 'none'}\n"
            "Update facts and uncertainties briefly. "
            "Reply as JSON with keys facts (dict), uncertainties (list), history_summary (string, <=2 sentences), meta (dict with exec_path/notes/status). "
            "JSON only, no prose, keep it short (<=80 tokens)."
        )
        observer = Provider.build("belief_brf", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = observer.call(prompt)
        self._add_time("img_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        if isinstance(parsed, dict):
            facts = parsed.get("facts") or {}
            if isinstance(facts, dict):
                self.belief_state.facts.update(facts)

            uncertainties = parsed.get("uncertainties") or []
            if isinstance(uncertainties, list):
                self.belief_state.uncertainties = [str(u) for u in uncertainties if str(u).strip()]

            history = parsed.get("history_summary")
            if isinstance(history, str) and history.strip():
                self.belief_state.history_summary = history.strip()

            meta = parsed.get("meta") or parsed.get("meta_notes") or {}
            if isinstance(meta, dict):
                self.belief_state.meta.update(meta)

        self.main_memory.append_message(Role.USER, f"[Belief] {self._format_belief_state(compact=True)}", Type.REFLECT)

    def _bdi_generate_options(self) -> List[Commitment]:
        if not self.belief_state:
            return []

        exploration_hint = ""
        if config.ACTIVE_CONFIG.agent.action is config.ActionType.BDI_EXPLORATION:
            exploration_hint = (
                "If uncertainties block progress, add one exploratory option (e.g., expand view, inspect new objects)."
            )

        prompt = (
            f"Belief-State:\n{self._format_belief_state()}\n"
            f"Goal: {self.goal}\n"
            f"{exploration_hint}\n"
            "Generate up to three next commitments as a JSON list. "
            "Fields: subgoal, action_type, success_criteria, max_exec_steps (1-4), reason. "
            "Focus on short, executable steps using the available primitive actions. "
            "Return compact JSON only, no prose, keep under 80 tokens."
        )
        planner = Provider.build("bdi_options", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = planner.call(prompt)
        self._add_time("img_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        options: List[Commitment] = []

        if isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, dict):
                    commit = self._commitment_from_dict(entry)
                    if commit:
                        options.append(commit)
        elif isinstance(parsed, dict):
            commit = self._commitment_from_dict(parsed)
            if commit:
                options.append(commit)

        return options

    def _bdi_filter_intention(self, options: List[Commitment]) -> Optional[Commitment]:
        if not options:
            return None

        exploration_toggle = ""
        if config.ACTIVE_CONFIG.agent.action is config.ActionType.BDI_EXPLORATION:
            exploration_toggle = "You may set exploration_mode to true when data is missing or repeated actions fail."

        options_payload = [
            {
                "index": idx,
                "subgoal": opt.subgoal,
                "success": opt.success_criteria,
                "reason": opt.reason,
                "max_exec_steps": opt.max_exec_steps,
            }
            for idx, opt in enumerate(options)
        ]

        prompt = (
            f"Belief snapshot: {self._format_belief_state(compact=True)}\n"
            f"Options: {json.dumps(options_payload, ensure_ascii=True)}\n"
            f"{exploration_toggle}\n"
            "Choose the best option. Reply as JSON {\"index\": <int>, \"exploration_mode\": <bool>, \"note\": \"short reason\"}. "
            "JSON only, no prose, keep under 40 tokens."
        )
        selector = Provider.build("bdi_filter", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = selector.call(prompt)
        self._add_time("img_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        choice_idx: int = 0
        exploration_flag: Optional[bool] = None

        if isinstance(parsed, dict):
            idx_val = parsed.get("index") if "index" in parsed else parsed.get("choice")
            try:
                choice_idx = int(idx_val)
            except (TypeError, ValueError):
                choice_idx = 0

            exploration_flag = parsed.get("exploration_mode")
            note = parsed.get("note")
            if note:
                self.main_memory.append_message(Role.USER, f"[Intent-Filter] {note}", Type.REFLECT)

        if exploration_flag is not None and config.ACTIVE_CONFIG.agent.action is config.ActionType.BDI_EXPLORATION:
            self.exploration_mode = bool(exploration_flag)

        if choice_idx < 0 or choice_idx >= len(options):
            choice_idx = 0

        return options[choice_idx]

    def _bdi_execute_commitment(self, commitment: Commitment, perception: str) -> None:
        tool_selection = [ToolGroup.ENV]
        if self.exploration_mode:
            tool_selection.append(ToolGroup.MEM)

        imagination_task = (
            f"Commitment: {commitment.subgoal}\n"
            f"Action type: {commitment.action_type}\n"
            f"Success criteria: {commitment.success_criteria}\n"
            f"Belief state (short): {self._format_belief_state(compact=True)}\n"
            f"Plan a minimal sequence (1 to {commitment.max_exec_steps} primitive tool calls) that reaches the criterion.\n"
            "- Focus ONLY on this commitment, not the whole goal.\n"
            "- If out of range, include the needed move then the action; otherwise just the action.\n"
            "- Return a compact list/line of the exact tool calls (max "
            f"{commitment.max_exec_steps}), no exec_done, no summaries."
        )

        visible_ids = self._perception_ids(perception)
        allowed_ids = ", ".join(sorted(set(visible_ids))) if visible_ids else "none-visible"

        realization_context = (
            f"{_BDI_POLICY_MESSAGE}\n"
            f"Commitment: {commitment.subgoal}\n"
            f"Success: {commitment.success_criteria}\n"
            f"Planned action(s): {{imagination}}\n"
            f"Perception: {perception}\n"
            f"Allowed object ids: {allowed_ids}\n"
            f"Exploration_Mode: {self.exploration_mode}. "
            "Execute ONLY the planned tool calls, in order, using primitive actions. "
            "Do not add extra steps or exec_done flags. If out of range, move first. "
            "Use only IDs from Allowed object ids; if the target is absent, move/observe first instead of inventing one. "
            "Respond with tool calls only, executable order. No prose, no summaries."
        )

        imagination_filter: Optional[Callable[[str], str]] = None
        if self.use_bdi and config.ACTIVE_CONFIG.agent.reflect is not config.ReflectType.OFF:
            imagination_filter = lambda plan: self._apply_action_critic(
                plan,
                commitment=commitment,
                perception=perception,
                allowed_ids=visible_ids,
            )

        self._imaginator_realisator_step(
            imagination_task=imagination_task,
            realization_context=realization_context,
            tools=tool_selection,
            name="bdi_action",
            imagination_filter=imagination_filter,
        )

    def _generate_ranked_actions(self, perception: str) -> list[ActionCandidate]:
        allowed_ids = self._perception_ids(perception)
        names, inventory_ids = self._perception_lookup(perception)
        agent_pos, pos_by_id = self._parse_positions(perception)
        allowed_serialized = ", ".join(sorted(set(allowed_ids))) if allowed_ids else "none-visible"

        prompt = (
            f"Goal: {self.goal}\n"
            f"Perception: {perception}\n"
            f"Allowed object ids: {allowed_serialized}\n"
            "Generate 3 to 5 candidate primitive tool calls the agent could execute next to advance the goal.\n"
            "Each candidate must include: call (exact primitive tool call) and reason (why this is useful). "
            "Use only primitive calls: move_to_object <ID>, move_to_position <x> <y>, "
            "take_from <WHAT_ID> <FROM_ID|FLOOR>, drop_to <WHAT_ID> <TO_ID|FLOOR>, "
            "interact_with_object <ID> <OPERATOR>, interact_with_object_using_item <ID> <ITEM_ID> <OPERATOR>. "
            "Only use IDs from Allowed object ids or the inventory. "
            "Return compact JSON: {\"actions\": [{\"call\": \"...\", \"reason\": \"...\"}, ...]}. JSON only."
        )

        generator = Provider.build("ranked_action_options", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = generator.call(prompt)
        self._add_time("plan_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        raw_actions: list[Any] = []
        if isinstance(parsed, dict):
            raw_actions = parsed.get("actions") or parsed.get("options") or parsed.get("candidates") or []
        elif isinstance(parsed, list):
            raw_actions = parsed

        candidates: list[ActionCandidate] = []
        for entry in raw_actions:
            call_raw = ""
            reason = ""
            if isinstance(entry, dict):
                call_raw = str(entry.get("call") or entry.get("action") or entry.get("toolcall") or "").strip()
                reason = str(entry.get("reason") or entry.get("why") or "").strip()
            elif isinstance(entry, str):
                call_raw = entry.strip()

            if not call_raw:
                continue

            normalized_list = self._deterministic_action_critic(
                [call_raw],
                allowed_ids=allowed_ids,
                inventory_ids=inventory_ids,
                names=names,
                agent_pos=agent_pos,
                pos_by_id=pos_by_id,
            )
            if not normalized_list:
                continue

            normalized_call = normalized_list[0]
            if any(c.call == normalized_call for c in candidates):
                continue

            candidates.append(ActionCandidate(call=normalized_call, reason=reason))

        return candidates

    def _select_ranked_action(
        self, candidates: list[ActionCandidate], perception: str, feedback: str
    ) -> tuple[int, str]:
        payload = [
            {"index": idx, "call": candidate.call, "reason": candidate.reason}
            for idx, candidate in enumerate(candidates)
        ]
        prompt = (
            f"Goal: {self.goal}\n"
            f"Perception: {perception}\n"
            f"Recent feedback: {feedback or 'none'}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=True)}\n"
            "Choose the best next action. "
            "Briefly simulate the immediate consequences (2-3 steps) to avoid dead-ends. "
            "Reply as JSON {\"index\": <int>, \"why\": \"short rationale\"}. JSON only."
        )

        selector = Provider.build("ranked_action_selector", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = selector.call(prompt)
        self._add_time("trial_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        idx = 0
        reason = ""
        if isinstance(parsed, dict):
            try:
                idx = int(parsed.get("index") if "index" in parsed else parsed.get("choice") or 0)
            except (TypeError, ValueError):
                idx = 0

            reason_val = parsed.get("why") or parsed.get("reason") or parsed.get("note")
            if isinstance(reason_val, str):
                reason = reason_val.strip()

        if idx < 0 or idx >= len(candidates):
            idx = 0

        return idx, reason

    def _execute_ranked_action(self, call: str, perception: str, selection_reason: str) -> None:
        context = (
            f"{_BDI_POLICY_MESSAGE}\n"
            f"Goal: {self.goal}\n"
            f"Chosen action: {call}\n"
            f"Choice rationale: {selection_reason or 'best fit for goal'}\n"
            f"Perception: {perception}\n"
            "Execute exactly the chosen tool calls, in order. "
            "Respond only with tool calls. Do not invent extra steps."
        )

        realisator = ToolProvider.build("ranked_action_realisator", self.realisator_model, Memory(context))
        register_tools(realisator, ToolGroup.ENV)

        base_result_len = len(Resultbuffer.buffer)
        start = time.time()
        realisator.invoke(context)
        self._add_time("action_time_s", time.time() - start)

        has_error, errors = process_formal_errors(realisator.memory, collect=True)
        performed_action = any(
            isinstance(r, (ActionNotPossible, Success)) for r in Resultbuffer.buffer[base_result_len:]
        )

        if not performed_action and not has_error:
            errors.append(
                {
                    "agent_message": "No tool call executed. Repeat the chosen action exactly as given.",
                    "hint": "Return only tool calls with concrete object ids; avoid prose.",
                    "context": None,
                }
            )
            FormalError(
                "Ranked action execution produced no tool calls.",
                console_message="Ranked action execution produced no tool calls.",
                hint="Return the selected tool call verbatim.",
            )

        if errors:
            process_formal_errors(self.main_memory)

    def _update_ranked_action_loop(self, perception: str) -> None:
        self.main_memory.append_message(Role.USER, perception, Type.PERCEPTION)

        feedback = self._bdi_collect_feedback()
        candidates = self._generate_ranked_actions(perception)
        if not candidates:
            self.main_memory.append_message(
                Role.USER,
                "No valid action candidates available for the current perception.",
                Type.FEEDBACK,
            )
            self.main_memory.save()
            return

        choice_idx, selection_reason = self._select_ranked_action(candidates, perception, feedback)
        chosen = candidates[choice_idx]
        log_reason = selection_reason or chosen.reason
        self.main_memory.append_message(
            Role.USER,
            f"[RankedChoice] {chosen.call} | reason: {log_reason}",
            Type.PLAN,
        )

        self._execute_ranked_action(chosen.call, perception, log_reason)
        self.had_action = True
        self.main_memory.save()

    def _react_generate_action(self, perception: str, feedback: str) -> tuple[str, str]:
        allowed_ids = self._perception_ids(perception)
        names, inventory_ids = self._perception_lookup(perception)
        agent_pos, pos_by_id = self._parse_positions(perception)
        allowed_serialized = ", ".join(sorted(set(allowed_ids))) if allowed_ids else "none-visible"

        prompt = (
            f"Goal: {self.goal}\n"
            f"Perception: {perception}\n"
            f"Recent feedback: {feedback or 'none'}\n"
            f"Allowed object ids: {allowed_serialized}\n"
            "Follow vanilla ReAct: First write a short Thought on what to do next, then pick exactly one primitive tool call. "
            "Allowed primitive tool calls: move_to_object <ID>, move_to_position <x> <y>, take_from <WHAT_ID> <FROM_ID|FLOOR>, "
            "drop_to <WHAT_ID> <TO_ID|FLOOR>, interact_with_object <ID> <OPERATOR>, interact_with_object_using_item <ID> <ITEM_ID> <OPERATOR>. "
            "Use only IDs from Allowed object ids or inventory. "
            "Respond as compact JSON: {\"thought\": \"...\", \"action\": \"<toolcall>\"}. JSON only."
        )

        generator = Provider.build("react_reason", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reply = generator.call(prompt)
        self._add_time("plan_time_s", time.time() - start)

        parsed = self._safe_json_parse(reply)
        thought = ""
        action_raw = ""

        if isinstance(parsed, dict):
            thought = str(parsed.get("thought") or parsed.get("reasoning") or parsed.get("analysis") or "").strip()
            action_raw = str(parsed.get("action") or parsed.get("toolcall") or parsed.get("call") or "").strip()

        if not action_raw and isinstance(reply, str):
            for line in reply.splitlines():
                if "action" in line.lower():
                    action_raw = line.split(":", 1)[-1].strip()
                if "thought" in line.lower() and not thought:
                    thought = line.split(":", 1)[-1].strip()

        normalized = self._deterministic_action_critic(
            [action_raw],
            allowed_ids=allowed_ids,
            inventory_ids=inventory_ids,
            names=names,
            agent_pos=agent_pos,
            pos_by_id=pos_by_id,
        )
        action = normalized[0] if normalized else ""
        return thought, action

    def _execute_react_action(self, action: str, perception: str, thought: str) -> None:
        context = (
            f"{_BDI_POLICY_MESSAGE}\n"
            f"Goal: {self.goal}\n"
            f"Thought: {thought or 'n/a'}\n"
            f"Action: {action}\n"
            f"Perception: {perception}\n"
            "Execute exactly the specified action using primitive tool calls. "
            "Respond only with tool calls, no prose. If out of range, include the move first."
        )

        realisator = ToolProvider.build("react_realisator", self.realisator_model, Memory(context))
        register_tools(realisator, ToolGroup.ENV)

        base_result_len = len(Resultbuffer.buffer)
        start = time.time()
        realisator.invoke(context)
        self._add_time("action_time_s", time.time() - start)

        has_error, errors = process_formal_errors(realisator.memory, collect=True)
        performed_action = any(
            isinstance(r, (ActionNotPossible, Success)) for r in Resultbuffer.buffer[base_result_len:]
        )

        if not performed_action and not has_error:
            errors.append(
                {
                    "agent_message": "No tool call executed. Repeat the chosen action exactly as given.",
                    "hint": "Return only tool calls with explicit ids.",
                    "context": None,
                }
            )
            FormalError(
                "ReAct execution produced no tool calls.",
                console_message="ReAct execution produced no tool calls.",
                hint="Return the selected tool call verbatim.",
            )

        if errors:
            process_formal_errors(self.main_memory)

    def _update_react_loop(self, perception: str) -> None:
        self.main_memory.append_message(Role.USER, perception, Type.PERCEPTION)

        feedback = self._bdi_collect_feedback()
        thought, action = self._react_generate_action(perception, feedback)
        if not action:
            self.main_memory.append_message(
                Role.USER,
                "ReAct could not produce a valid action for the current perception.",
                Type.FEEDBACK,
            )
            self.main_memory.save()
            return

        if thought:
            self.main_memory.append_message(Role.USER, f"[ReAct Thought] {thought}", Type.REFLECT)
        self.main_memory.append_message(Role.USER, f"[ReAct Action] {action}", Type.PLAN)

        self._execute_react_action(action, perception, thought)
        self.had_action = True
        self.main_memory.save()

    def update(self, perception: str):
        current.AGENT = self
        current.ENTITY = self.entity

        if self.use_bdi:
            self._update_bdi_loop(perception)
            current.AGENT = None
            current.ENTITY = None
            return

        if self.use_ranked_actions:
            self._update_ranked_action_loop(perception)
            current.AGENT = None
            current.ENTITY = None
            return

        if self.use_react:
            self._update_react_loop(perception)
            current.AGENT = None
            current.ENTITY = None
            return

        if not self.triggered_replan and self.had_action:
            t0 = time.time()
            self._reflect(perception) # stellt fest, was das ergebniss der aktion ist. ziel im focused plannode schon erreicht? dann in planung, sonst Frage: sind wir noch auf einem guten weg? ja -> weiter; nein -> nächste idee
            self._add_time("reflect_time_s", time.time() - t0)
            self.main_memory.save()
            self.had_action = False

        if self.triggered_replan or config.ACTIVE_CONFIG.agent.trial is config.TrialType.OFF:
            t0 = time.time()
            self._plan() # should set self.plan with a full Plan
            self._add_time("plan_time_s", time.time() - t0)
            self.main_memory.set_plan(self.plan)
            self.main_memory.save()
        
        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            pretty(bullet("PLAN: " + self.plan.to_string(), color=Color.MAGENTA))

        if not self.triggered_replan:
            t0 = time.time()
            self._observe(perception) # returns aufbereitete observation mit den für die aufgabe relevanten inhalte
            self._add_time("observe_time_s", time.time() - t0)
            self.main_memory.save()

        if not self.triggered_replan:
            t0 = time.time()
            self._trial() # generiert List[str] mit ideen um das ziel im focused plannode zu erreichen
            self._add_time("trial_time_s", time.time() - t0)
            self.main_memory.set_plan(self.plan)
            self.main_memory.save()

        if config.ACTIVE_CONFIG.agent.trial is not config.TrialType.OFF:
            pretty(bullet("TRIAL: " + self.plan.get_trial().to_string(), color=Color.RED))

        if not self.triggered_replan:
            t0 = time.time()
            self._act(perception) # führt beste idee aus
            self._add_time("action_time_s", time.time() - t0)
            self.main_memory.save()
            self.had_action = True

        current.AGENT = None
        current.ENTITY = None

    def _observe(self, perception: str):
        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.OFF:
            self.main_memory.append_message(Role.USER, perception, Type.OBSERVATION)
            return

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            active_plan = "goal"
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            active_plan = "current plan step"
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
            active_plan = "focused plan step"
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
            active_plan = "focused plan node"
        else:
            raise Exception()
        
        observer = Provider.build("observer", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        observation = observer.call(f"Perception: {perception}\nWhat do you observe? Make sure to verify facts. Respect facts, not assumptions. What is relevant for your {active_plan}? Only tell about new discouveries. (short)")
        self._add_time("img_time_s", time.time() - start)

        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.ON:
            self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)
        else:
            pass
            #self._memorize(observation, Type.OBSERVATION)

    def _trial(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return
        
        if config.ACTIVE_CONFIG.agent.trial is config.TrialType.OFF:
            return
        
        #havent tried an idea yet
        if not self.plan.get_trial().ideas:
            self._generate_trials()
            return

        completed = self._ask_yes_no(
            name="trial",
            context=(
                f"{self.main_memory.get_history()}"
            ),
            question="Is the Plan completed?",
        )

        if not completed:
            promising, ratio = self._ask_yes_no_with_rationale(
                name="trial",
                context=(
                    f"{self.main_memory.get_history()}"
                ),
                #question="Is the Plan still promising?",
                question="Is the plan still promising? Answer no when recent actions produced no new state change, no progress toward the focused plan step, or when the agent is repeating actions that do not modify the situation."
            )

            if promising:
                return

            self.triggered_replan = f"Failed to accomplish plan: {ratio}"
            return

        trials = self.plan.get_trial()

        trials.mark_current_completed()
        self.plan.mark_current_completed()

        if not trials.current_step():
            # Reset index so newly generated ideas start from the first new entry.
            trials.ideas = []
            trials.current_index = 0
            self._generate_trials()
            return

    def _generate_trials(self):
        pretty(banner("Generating trials"))
        if self.plan.get_trial().completed:
            self._imaginator_realisator_step(
                imagination_task=(
                    f"Plan:\n{self.plan.to_string(False)}\n"
                    " Brainstorm a few concrete ideas on how to achieve this task. But only approaches using actions the human can execute!"
                    " Avoid repeating previously tried approaches."
                ),
                realization_context=(
                    f"Plan:\n{self.plan.to_string(False)}\n"
                    f"Tried approaches: {self.plan.get_trial().completed}\n"
                    "Use add_trial(text) for each distinct idea to try. Keep them concise and actionable."
                    "Respond only with add_trial() tool calls, one per idea. If there is no new idea call noop()."
                ),
                tools=ToolGroup.TRIAL,
                name="trial",
            )
        else:
            self._imaginator_realisator_step(
                imagination_task=(
                    f"Plan:\n{self.plan.to_string(False)}\n"
                    "What would be the naive approach to this? Use only actions the human can execute!"
                    "Tell the simpel action to archive it."
                ),
                realization_context=(
                    f"Plan:\n{self.plan.to_string(False)}\n"
                    "Use add_trial(text) for each distinct idea to try. Keep them concise and actionable."
                    "Respond only with add_trial() tool calls, one per idea. If there is no new idea call noop()."
                ),
                tools=ToolGroup.TRIAL,
                name="trial",
            )

        if not self.plan.get_trial().current_step():
            if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
                plan = "current plan"
            elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
                plan = "focused plan step"
            elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
                plan = "focused plan node"
            else:
                raise Exception()
            self.triggered_replan = f"failed to get {plan} done"
        else:
            print(f"Current trial {self.plan.get_trial().current_step()}")
    
    def _act(self, perception: str):
        prompt = "Give best next action to perform (short answer, but explicit)."

        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But always tell the next action"

        if config.ACTIVE_CONFIG.agent.action in (config.ActionType.DIRECT, config.ActionType.DIRECT_RETRY):
            realisator = ToolProvider.build("actor", self.realisator_model, memory=self.main_memory)
            register_tools(realisator, ToolGroup.ENV)
            attempts = 3 if config.ACTIVE_CONFIG.agent.action is config.ActionType.DIRECT_RETRY else 1
            correction_suffix = ""
            final_has_error = False
            last_errors: list[dict] = []
            base_result_len = len(Resultbuffer.buffer)

            for attempt in range(attempts):
                instruction = (
                    perception
                    + ". "
                    + prompt
                    + "Use concise, executable actions. Your answer must only consist of toolcalls."
                )
                if correction_suffix:
                    instruction += f" {correction_suffix}"

                start = time.time()
                realisator.invoke(instruction)
                self._add_time("real_time_s", time.time() - start)

                has_error, errors = process_formal_errors(realisator.memory, collect=True)

                performed_action = any(
                    isinstance(r, (ActionNotPossible, Success)) for r in Resultbuffer.buffer[base_result_len:]
                )
                if not performed_action and not has_error:
                    has_error = True
                    errors.append(
                        {
                            "agent_message": "No tool call executed. Respond with explicit tool calls that carry out the action.",
                            "hint": "Return only tool calls with concrete object IDs; avoid prose.",
                            "context": None,
                        }
                    )

                final_has_error = has_error
                last_errors = errors

                if not has_error:
                    break

                hint_texts = [e.get("hint") for e in errors if e.get("hint")]
                agent_msgs = [e.get("agent_message") for e in errors if e.get("agent_message")]
                combined = "; ".join(hint_texts or agent_msgs)
                correction_suffix = (
                    f"Retry with these corrections: {combined}. Keep tool calls minimal."
                    if combined
                    else "Retry using valid tool calls with explicit object IDs."
                )

            if final_has_error:
                FormalError(
                    "Direct action generation failed after multiple retries.",
                    console_message="Direct action generation failed after multiple retries.",
                    hint="Review the latest hints and return minimal tool calls with valid IDs.",
                )

            process_formal_errors()  # clear any remaining formal errors
        else:
            tc_prompt =  """
            Give exactly the toolcalls that arise from the planned action.
            Use correct object id.
            Toolcall order matters.
            Your answer must only consist of toolcalls.
            """
            #If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a precice and short question.
            #Else toolcalls only!

            self._imaginator_realisator_step(
                imagination_task=prompt,
                realization_context=f"{perception} Reasoning: {{imagination}} Task: {tc_prompt}",
                tools=ToolGroup.ENV,
                name="action",
            )

    def _reflect(self, perception):
        results = ""
        for role, msg, msg_type in process_action_results():
            results += msg + "\n"

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            self.main_memory.append_message(Role.USER, results, Type.FEEDBACK)
            return

        reflector = Provider.build("reflector", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        reflection = reflector.call(f"Result: {results}. \nReflect what effect the performed Actions had. What are the facts? What are assumptions about the new state? Wich assumptions that were made proved right or wrong? (short)")
        self._add_time("img_time_s", time.time() - start)

        self.main_memory.append_message(role=Role.USER, message=reflection, type=Type.REFLECT)

        #if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.ON:
        #    self.main_memory.append_message(Role.USER, reflection, Type.REFLECT)
        #else:
        ##    self._memorize(reflection)

    def _plan(self):
        if self.triggered_replan:
            self._replan(self.triggered_replan)
            return

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            plan = "current plan"
            active_plan_completed = f"{plan} completed and the goal reached"
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
            plan = "focused plan step"
            active_plan_completed = f"{plan} completed"
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
            plan = "focused plan node"
            active_plan_completed = f"{plan} completed"
        else:
            raise Exception()

        ans, rationale = self._ask_yes_no_with_rationale(
            "planner",
            context = self.main_memory.get_history(),
            question = f"Based on the current reflection of the last action, is the {active_plan_completed}?",
        )

        if ans is True:
            print(f"The {plan} is completed because {rationale}")
            self._next_plan_part()
        else:
            ans2, rationale2 = self._ask_yes_no_with_rationale(
                "planner",
                context = self.main_memory.get_history(),
                question = f"The {plan} is not completed because: {rationale}. Does it still seem promising?",
            )

            if ans2 is False:
                self.triggered_replan = rationale2
            else:
                print(f"The {plan} is still promising because {rationale2}")
                return #go on

    def _next_plan_part(self):
        next = self.plan.mark_current_completed()

        if next:
            return
        
        ans, rationale = self._ask_yes_no_with_rationale(
            "planner",
            context = self.main_memory.get_history(),
            question = f"Based on the current reflection of the last action, is the initial main goal reached?",
        )

        if ans:
            self.finished_reason = rationale
            return
        else:
            self._replan("Initial main goal is not yet reached")

    def _replan(self, reason):
        print(f"Replanning because: {reason}")

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            self.plan = self._plan_free(reason)
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
            self.plan = self._plan_steps(reason)
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
            self.plan = self._plan_tree(reason)
        else:
            raise Exception()

        self.main_memory.set_plan(self.plan)
        self.triggered_replan = None
                

    def _plan_free(self, replan_reason=None) -> FreePlan:
        if replan_reason:
            replan = f"Replanning because {replan_reason}\n"
        else:
            replan = ""

        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        plan_text = planner.call(
            f"""Main goal: {self.goal}\n{replan}
            Based on the current contex: Create a structured plan to reach the goal.
            """
        )
        self._add_time("img_time_s", time.time() - start)

        return FreePlan(goal=self.goal, description=plan_text)

    def _plan_steps(self, replan_reason=None) -> StepPlan:
        step_plan = StepPlan(goal=self.goal)
        self.plan = step_plan

        if replan_reason:
            replan = f"Replanning because {replan_reason}\n"
        else:
            replan = ""

        self._imaginator_realisator_step(
            imagination_task=(f"Main goal: {self.goal}\n{replan}Create a new strucutred plan to reach the goal."),
            realization_context=(f"Plan: {{imagination}}\nUse toolcalls to add the steps specified in the plan for the user."),
            tools=ToolGroup.PLAN,
            name="plan"
        )

        return step_plan
    def _plan_tree(self, replan_reason=None) -> TreePlan:
        if replan_reason:
            replan = f"Replanning because {replan_reason}\n"
        else:
            replan = ""

        self.plan = TreePlan.new(self.goal)
        self.main_memory.set_plan(self.plan)

        self._decompose_tree_plan()

        active_node = self._choose_active_leaf()
        if active_node is None:
            raise Exception("No available plan nodes to execute")

        if isinstance(self.plan, TreePlan):
            self.plan.focus = active_node
            plan_overview = self.plan.format_full()
            self.main_memory.set_plan(
                "FULL PLAN TREE:\n" + plan_overview +
                f"\nCurrent target: [{active_node.id}] {active_node.data}"
            )

        return self.plan
    
    def _choose_active_leaf(self) -> "PlanNode | None":
        leaves = self.plan.leaf_nodes()
        if not leaves:
            return None

        current_focus = self.plan.focus if isinstance(self.plan, TreePlan) and self.plan.focus in leaves else None

        if current_focus:
            should_keep = self._ask_yes_no(
                name = "active_leaf",
                context=(
                    f"Goal: {self.goal}\nPlan tree:\n{self.plan.format_full()}"
                ),
                question="Is the current focus still the best leaf task to pursue next?",
            )
            if should_keep:
                return current_focus

        chooser = Provider.build("leaf_selector", self.imaginator_model, memory=self.main_memory)
        leaf_listing = "\n".join([f"[{leaf.id}] {leaf.data}" for leaf in leaves])
        start = time.time()
        choice = chooser.call(
            f"Goal: {self.goal}\nPlan tree:\n{self.plan.format_full()}\n"
            f"Available leaf tasks:\n{leaf_listing}\n"
            "Select the best suited leaf task by replying with its id and a short justification.",
        )
        self._add_time("img_time_s", time.time() - start)

        selected_leaf = None
        for leaf in leaves:
            if str(leaf.id) in choice:
                selected_leaf = leaf
                break

        if selected_leaf is None:
            selected_leaf = leaves[0]

        if isinstance(self.plan, TreePlan):
            self.plan.focus = selected_leaf
        return selected_leaf

    def _decompose_tree_plan(self):
        for _ in range(0, 3):
            current_plan = self.plan.format_full()
            is_ready = self._ask_yes_no(
                name="decompose",
                context=(
                    f"Goal: {self.goal}\nCurrent plan tree:\n{current_plan}\n"
                ),
                question="Is the current plan tree sufficiently decomposed to start execution?",
            )

            if is_ready:
                return

            self._imaginator_realisator_step(
                imagination_task=(
                    f"Goal: {self.goal}\nCurrent plan tree:\n{current_plan}\n"
                    "Study the current plan tree and decide how to refine it."
                    " Suggest concise updates to decompose or (in rare cases) prune nodes before execution."
                ),
                realization_context=(
                    f"""Goal: {self.goal}\nPlan tree:\n{current_plan}\n
                    Use decompose_node(task_node_id, sub_nodes) to split tasks, delete_node(task_node_id, delete_children)
                     to remove or merge tasks, mark_done(task_node_id) to label completed nodes
                     mark_focued(task_node_id) to set the next focus. Use only the provided plan editing tools to update the plan tree.
                    Update like this: {{imagination}}
                    """
                ),
                tools=ToolGroup.DECOMPOSE,
                name="decompose",
            )

    def _imaginator_realisator_step(
        self,
        imagination_task: str,
        realization_context: str,
        tools: ToolGroup,
        name: str,
        imagination_filter: Optional[Callable[[str], str]] = None,
    ) -> None:
        imaginator = Provider.build(name + "_imaginator", self.imaginator_model, memory=self.main_memory)

        start_imagination = time.time()
        imagination = imaginator.call(imagination_task)
        self._add_time("img_time_s", time.time() - start_imagination)

        if imagination_filter:
            imagination = imagination_filter(imagination)
            if not imagination or not str(imagination).strip():
                FormalError(
                    "Action generation aborted: critic removed all tool calls.",
                    console_message="No valid tool calls after validation; replan or regenerate actions.",
                    hint="Revise the plan or perception, then regenerate a concrete tool call.",
                )
                return

        realisator = ToolProvider.build(
            name + "_realisator",
            self.realisator_model,
            Memory(realization_context),
        )
        register_tools(realisator, tools)

        correction_suffix = ""
        final_has_error = False
        last_error_payloads: list[dict] = []
        base_result_len = len(Resultbuffer.buffer)

        for attempt in range(0, 3):
            ctx = realization_context.replace("{imagination}", imagination) + correction_suffix
            start_real = time.time()
            reply = realisator.invoke(ctx)
            self._add_time("real_time_s", time.time() - start_real)

            has_error, errors = process_formal_errors(realisator.memory, collect=True)

            needs_action = False
            if not has_error:
                has_env_tools = False
                if isinstance(tools, ToolGroup):
                    has_env_tools = tools is ToolGroup.ENV or tools is ToolGroup.ALL
                else:
                    has_env_tools = ToolGroup.ENV in tools or ToolGroup.ALL in tools

                if has_env_tools:
                    new_results = Resultbuffer.buffer[base_result_len:]
                    performed_action = any(
                        isinstance(r, (ActionNotPossible, Success)) for r in new_results
                    )
                    if not performed_action:
                        needs_action = True

            if needs_action:
                has_error = True
                errors = errors or []
                errors.append(
                    {
                        "agent_message": "No tool call executed. Respond with explicit tool calls that carry out the planned actions.",
                        "hint": "Return only tool calls in executable order. Move into range first when needed.",
                        "context": None,
                    }
                )

            final_has_error = has_error
            last_error_payloads = errors

            if not has_error:
                break

            if not current.any_action():
                correction_suffix = "If nothing is todo use the 'noop'"

            hint_texts = [e.get("hint") for e in errors if e.get("hint")]
            agent_msgs = [e.get("agent_message") for e in errors if e.get("agent_message")]
            combined = "; ".join(hint_texts or agent_msgs)
            correction_suffix = (
                f" Retry with these corrections: {combined}. Keep tool calls minimal."
                if combined
                else " Retry using valid tool calls with explicit object IDs."
            )

            if config.ACTIVE_CONFIG.agent.action is config.ActionType.IMG_RETRY:
                self.main_memory.append_message(
                    Role.USER,
                    "Last step was not specified well. Please provide more explicit instructions",
                )
                break

            #if config.ACTIVE_CONFIG.agent.action is config.ActionType.IMG_QUESTION:
            #    if "Question" in reply:
            #        correction_suffix = imaginator.invoke(
            #            reply + " Use explicit object IDs and absolute positions.",
            #            imagination_task + " Provide an explicit and precise instruction.",
            #            append=False,
            #        )
            #    else:
            #        break

        if final_has_error:
            hint_texts = [e.get("hint") for e in last_error_payloads if e.get("hint")]
            agent_msgs = [e.get("agent_message") for e in last_error_payloads if e.get("agent_message")]
            FormalError(
                "Action generation failed after multiple retries.",
                console_message=(
                    "Action generation failed after multiple retries. "
                    "Review the latest hints and adjust the prompt."
                ),
                hint="No action executed",
            )

        process_formal_errors() #clear errors

        return


        #def _memorize(self, context: str, task: str):
    #        imagination = self._imaginator_realisator_step(
    #            imagination_task=(
    #            f"""
    #            Is there something important to memorize from the data? If so tell short and precise what it is. Only information that is useful for the Plan.
    #            Are there obsolete or redundat MEMORIES that need to be deleted? Keep only important ones.
    #            \n:{context}
    #            \n{task}
    #            """
    #        ),
    #        realization_context=(
    #            f"Memories: {str(self.main_memory.memories)}\n"
    #            f"{context}\nReasoning: {{imagination}}\n"
    #            "Use the tools to store and/or delete memories like told. If there is nothing to do use the noop tool."
    #        ),
    #        tools=ToolGroup.MEM,
    #        realisator_system="Respond by calling the given functions only!",
    #        imaginator_name="memo_imaginator",
    #        realisator_name="memo_realisator",
    #        allow_question_retry=False,
    #    )
#
#          return imagination

    def _ask_yes_no(self, name: str, context: str, question: str) -> bool:
        current.ANSWER_BUFFER = None

        self._imaginator_realisator_step(
                imagination_task=f"{context}\nQuestion: {question}\nProvide a short reasoning before deciding.",
                realization_context=f"{context}\nQuestion: {question}\nReasoning: {{imagination}}\nRespond by calling yes() or no() to answer the question.",
                tools=ToolGroup.QA,
                name=name+"_qa",
        )

        return current.get_answer()
    
    def _ask_yes_no_with_rationale(self, name: str, context: str, question: str) -> Tuple[bool, str]:
        current.ANSWER_BUFFER = None

        self._imaginator_realisator_step(
                imagination_task=f"{context}\nQuestion: {question}\nProvide a short reasoning before deciding.",
                realization_context=f"{context}\nQuestion: {question}\nReasoning: {{imagination}}\nRespond by calling yes or no with the rationale to answer the question.",
                tools=ToolGroup.QA_RATIO,
                name=name+"_qa_rationale",
        )

        return current.get_answer(), current.get_rationale()
