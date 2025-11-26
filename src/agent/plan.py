#class Plan:
#    #TODO either string, list[strings] or TREE depending on plan type

class Plan():
    init(goal)
    def to_String()
        return goal

class FreePlan(Plan):
    goal + frertextplan to reach it

    def abstract set()
    pass

class StepPlan(APlanBC):
    def abstract append()
    def abstract mark_complete(id)
    def abstract delete()
    pass

#class PlanNode:
#    #TODO take from supermem

class TreePlan(Plan):
    #TODO
    root_node
    pass


def _format_plan_tree(self, node=None, prefix: str = "", active_node=None) -> str:
        node = node or self.main_memory.plan_root
        active_node = active_node or self.main_memory.plan_node

        markers = []
        if node.done:
            markers.append("done")
        if active_node and node.id == active_node.id:
            markers.append("current focus")
        marker = f" ({'; '.join(markers)})" if markers else ""
        id_label = self._node_identifier(node)
        lines = [f"{prefix}- {id_label} {node.data}{marker}"]

        for child in node.children:
            lines.append(self._format_plan_tree(child, prefix + "  ", active_node))

        return "\n".join(lines)

    def _node_identifier(self, node) -> str:
        return f"[{node.id}]"

    def _leaf_nodes(self, node=None) -> list:
        node = node or self.main_memory.plan_root

        if node.done:
            return []

        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._leaf_nodes(child))

        return leaves