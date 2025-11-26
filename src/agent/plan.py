from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Plan:
    goal: str

    def to_string(self) -> str:
        return self.goal

    def mark_current_completed(self):
        """Mark the current plan item as completed.

        Returns a truthy value if there is more work to do; otherwise falsy.
        """

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class FreePlan(Plan):
    description: str

    def to_string(self) -> str:
        return f"Goal: {self.goal}\nPlan: {self.description}"


@dataclass
class StepPlan(Plan):
    steps: List[str] = field(default_factory=list)
    current_index: int = 0
    completed: List[str] = field(default_factory=list)

    def add_step(self, step: str) -> None:
        self.steps.append(step)

    def current_step(self) -> Optional[str]:
        if self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    def mark_current_completed(self) -> Optional[str]:
        if self.current_index < len(self.steps):
            self.completed.append(self.steps[self.current_index])
            self.current_index += 1
        return self.current_step()

    def to_string(self) -> str:
        completed_text = "\n".join([f"- [x] {s}" for s in self.completed])
        remaining_text = "\n".join(
            [f"- [ ] {s}" for s in self.steps[self.current_index :]]
        )
        sections = [f"Goal: {self.goal}"]
        if completed_text:
            sections.append("Completed steps:\n" + completed_text)
        if remaining_text:
            sections.append("Upcoming steps:\n" + remaining_text)
        return "\n\n".join(sections)


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


@dataclass
class TreePlan(Plan):
    root: PlanNode
    focus: PlanNode
    plan_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)

    @classmethod
    def new(cls, goal: str) -> "TreePlan":
        root = PlanNode(goal)
        return cls(goal=goal, root=root, focus=root)

    def add_step(self, step: str) -> None:
        self.plan_steps.append(step)

    def mark_current_completed(self) -> Optional[PlanNode]:
        self.mark_node_done(self.focus.id)
        next_focus = self._first_open_leaf()
        if next_focus:
            self.focus = next_focus
        return next_focus

    def decompose_node(self, task_node_id: int, sub_nodes: List[str]) -> PlanNode:
        node = self.root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        for sub_task in sub_nodes:
            node.add_child(sub_task)

        return node

    def delete_node(self, task_node_id: int, delete_children: bool = True) -> None:
        node = self.root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        node.remove(delete_children=delete_children)
        if self.focus and self.focus.id == task_node_id:
            self.focus = self.root

    def mark_node_done(self, task_node_id: int) -> PlanNode:
        node = self.root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        node.done = True
        completed_entry = f"[{node.id}] {node.data}"
        if completed_entry not in self.completed_steps:
            self.completed_steps.append(completed_entry)

        return node

    def mark_node_focus(self, task_node_id: int) -> PlanNode:
        node = self.root.find(task_node_id)
        if node is None:
            raise KeyError(f"Plan node with id {task_node_id} not found")

        self.focus = node
        return node

    def _first_open_leaf(self) -> Optional[PlanNode]:
        leaves = self.leaf_nodes()
        return leaves[0] if leaves else None

    def leaf_nodes(self, node: Optional[PlanNode] = None) -> list[PlanNode]:
        node = node or self.root

        if node.done:
            return []

        if not node.children:
            return [node]

        leaves: list[PlanNode] = []
        for child in node.children:
            leaves.extend(self.leaf_nodes(child))

        return leaves

    def format_plan_tree(
        self, node: Optional[PlanNode] = None, prefix: str = "", active_node: Optional[PlanNode] = None
    ) -> str:
        node = node or self.root
        active_node = active_node or self.focus

        markers = []
        if node.done:
            markers.append("done")
        if active_node and node.id == active_node.id:
            markers.append("current focus")
        marker = f" ({'; '.join(markers)})" if markers else ""
        id_label = self._node_identifier(node)
        lines = [f"{prefix}- {id_label} {node.data}{marker}"]

        for child in node.children:
            lines.append(self.format_plan_tree(child, prefix + "  ", active_node))

        return "\n".join(lines)

    def _node_identifier(self, node: PlanNode) -> str:
        return f"[{node.id}]"

    def to_string(self) -> str:
        return self.format_plan_tree(active_node=self.focus)

    def clone(self) -> "TreePlan":
        root_clone = self.root.clone(None)
        focus_clone = root_clone.find(self.focus.id) if self.focus else root_clone
        return TreePlan(
            goal=self.goal,
            root=root_clone,
            focus=focus_clone,
            plan_steps=list(self.plan_steps),
            completed_steps=list(self.completed_steps),
        )
