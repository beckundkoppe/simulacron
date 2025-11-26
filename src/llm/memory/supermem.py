import copy
from dataclasses import dataclass, field
from typing import List, Tuple
import config
from llm.memory.memory import Memory, Role, Type


@dataclass
class PlanNode:
    """A simple tree structure to represent nested plans and tasks."""

    data: str
    parent: "PlanNode | None" = None
    id: int | None = None
    children: list["PlanNode"] = field(default_factory=list)
    done: bool = False

    _counter: int = 0

    def __post_init__(self):
        if self.id is None:
            self.id = PlanNode._counter
        PlanNode._counter = max(PlanNode._counter + 1, (self.id or 0) + 1)

        if self.parent:
            self.parent.children.append(self)

    def add_child(self, data: str) -> "PlanNode":
        return PlanNode(data, parent=self)

    def find(self, search_id: int) -> "PlanNode | None":
        if self.id == search_id:
            return self

        for child in self.children:
            found = child.find(search_id)
            if found:
                return found

        return None

    def remove(self, delete_children: bool = True) -> None:
        if self.parent is None:
            raise ValueError("Cannot delete the root plan node")

        parent_children = self.parent.children
        if delete_children:
            self.parent.children = [child for child in parent_children if child is not self]
            return

        self.parent.children = [child for child in parent_children if child is not self]
        for child in self.children:
            child.parent = self.parent
            self.parent.children.append(child)
        self.children.clear()

    def clone(self, parent: "PlanNode | None" = None) -> "PlanNode":
        clone_node = PlanNode(self.data, parent=parent, id=self.id, done=self.done)
        for child in self.children:
            child.clone(clone_node)
        return clone_node

class SuperMemory(Memory):
    def __init__(self, goal, path) -> None:
        super().__init__(goal, path)

        #goal
        #_history
        self._learnings: List[str] = []
        self._plans: List[str] = []
        self.plan_root = PlanNode(goal)
        self.plan_node = self.plan_root
        self.memories: dict[int, str] = {}
        self._current_observation: str = []
        self.plan = None
        self.plan_steps: list[str] = []
        self.completed_steps: list[str] = []

    def add_plan(self, plan: str):
        self.plan = plan
        self.save()

    def add_learing(self, learn):
        self._learnings.append(learn)
        self.save()

    def mark_completed(self):
        if self.plan_steps:
            self.completed_steps.append(self.plan_steps.pop(0))

    def store_permanent_memory(self, information: str) -> int:
        new_id = max(self.memories.keys(), default=-1) + 1
        self.memories[new_id] = information
        return new_id

    def delete_permanent_memory(self, memory_id: int) -> None:
        if memory_id not in self.memories:
            raise KeyError(f"Memory id {memory_id} not found")
        del self.memories[memory_id]

    def decompose_plan_node(self, task_node_id: int, sub_nodes: List[str]) -> PlanNode:
        node = self.plan_root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        for sub_task in sub_nodes:
            node.add_child(sub_task)

        return node

    def delete_plan_node(self, task_node_id: int, delete_children: bool = True) -> None:
        node = self.plan_root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        node.remove(delete_children=delete_children)

    def mark_plan_node_done(self, task_node_id: int) -> PlanNode:
        node = self.plan_root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        node.done = True
        completed_entry = f"[{node.id}] {node.data}"
        if completed_entry not in self.completed_steps:
            self.completed_steps.append(completed_entry)

        return node

    def mark_plan_node_focus(self, task_node_id: int) -> PlanNode:
        node = self.plan_root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        self.plan_node = node
        return node

    def _get_history(self) -> List[Tuple[Type, Role, str]]:
        history_out = []

        ##### GOAL #####
        history_out.append((Type.GOAL, Role.SYSTEM, self._goal))
        ################

        COUNT = 2
        n = COUNT

        # find newest observation in the whole history
        current_observation = None
        recent_observations = []

        if self._history:
            elem = self._history.copy().pop()
            t,r,d = elem
            if t == Type.PERCEPTION:
                current_observation = elem


        for x in reversed(self._history):
            t, r, d = x
            recent_observations.append(x)
            if t == Type.PERCEPTION:
                n = n - 1
            if n <= 0:
                break
        
        recent_observations.reverse()

        mini = []
        for x in self._history:
            if x not in recent_observations and not current_observation:  # exclude the recent observations
                t, r, d = x
                if t != Type.PERCEPTION:
                    mini.append(x)

        ##### MINI HISTORY #####
        if mini:
            history_out.extend(mini)
        ########################

        if current_observation:
            recent_observations.remove(current_observation)

        ##### FULL HISTORY #####
        if recent_observations:
            history_out.extend(recent_observations)
        ########################

        ##### CURRENT OBSERVATION #####
        if current_observation:
            type, role, data = current_observation
            history_out.append((Type.CURRENT_OBSERVATION ,role,"CURRENT OBSERVATION: " + data))
        ###############################

        #learning

        ##### PLAN #####
        if self.completed_steps:
            history_out.append((Type.PLAN, Role.USER, "COMPLETED SUB-GOALS: "+ str(self.completed_steps)))

        if self.plan:
            history_out.append((Type.PLAN, Role.USER, self.plan))
        ################

        #memories

        return history_out

    def get_history(self) -> List[dict[str, str]]:
        max_allowed = int(config.Backend.n_context - max(500, config.Backend.n_context * 0.5))

        print(f"mem: {self.get_token_count()}/{max_allowed} ({config.Backend.n_context})")
        self.assure_max_token_count(max_allowed)
        history_out = self._get_history()

        return super()._build_history(history_out)
    
    def _store(self, path: str) -> None:
        history_out = self._get_history()
        self._save(path, history_out)

    def get_token_count(self) -> int:
        history_out = self._get_history()
        return Memory._approximate_token_count(str(history_out))
    
    def copy(self) -> "SuperMemory":
        new_copy = SuperMemory(self._goal, self.path)
        new_copy._history = copy.deepcopy(self._history)
        new_copy.plan = copy.deepcopy(self.plan)
        new_copy._learnings = copy.deepcopy(self._learnings)
        new_copy.memories = copy.deepcopy(self.memories)
        new_copy._current_observation = copy.deepcopy(self._current_observation)
        new_copy.plan_steps = copy.deepcopy(self.plan_steps)
        new_copy.completed_steps = copy.deepcopy(self.completed_steps)
        new_copy.plan_root = self.plan_root.clone(None)
        new_copy.plan_node = new_copy.plan_root.find(self.plan_node.id) if self.plan_node else new_copy.plan_root
        return new_copy
