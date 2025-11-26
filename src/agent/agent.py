from agent.helper import process_action_results, process_formal_errors
from agent.toolpool import ToolGroup, register_tools
import config
import current
from enviroment.entity import Entity
from enviroment.resultbuffer import FormalError
from llm.memory.memory import Memory, Role, Type
from llm.memory.supermem import PlanNode, SuperMemory
from llm.model import Model
from llm.provider import Provider
from llm.toolprovider import ToolProvider
from util.console import Color, bullet, pretty, title

class Agent:
    def __init__(self, goal: str, entity: Entity, imaginator_model: Model, realisator_model: Model):
        self.goal = goal
        self.entity = entity
        self.imaginator_model = imaginator_model
        self.realisator_model = realisator_model

        self.goal = goal
        self.main_memory = SuperMemory(goal=goal, path="main_memory.txt")

    def update(self, perception: str):
        current.AGENT = self
        current.ENTITY = self.entity

        self.plan()
        self.main_memory.save()

        plan_tree = self._format_plan_tree()
        pretty(title("Plan Tree", color=Color.CYAN))
        for line in plan_tree.split("\n"):
            pretty(bullet(line, color=Color.CYAN))

        self.observe(perception)
        self.main_memory.save()

        pretty(title("The current Plan", color=Color.MAGENTA))
        for step in self.main_memory.completed_steps:
            pretty(bullet(step, color=Color.RESET))
        for step in self.main_memory.plan_steps:
            pretty(bullet(step, color=Color.MAGENTA))

        self.act(perception)
        self.main_memory.save()

        self.reflect(perception)
        self.main_memory.save()

        current.AGENT = None
        current.ENTITY = None

    def observe(self, observation: str):
        self.main_memory.append_message(Role.USER, observation, Type.PERCEPTION)

        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.OFF:
            return
        
        mem = Memory()
        mem.append_message(Role.USER, observation)
        observer = Provider.build("observer", self.imaginator_model, memory=mem)
        observation = observer.call("What do you observe? What is interesting? (short)")
        self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)

        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.MEMORIZE:
            self.memorize(observation, "Do you now this room already?")

    def memorize(self, context: str, task: str):
        imagination = self._imaginator_realisator_step(
            imagination_task=(
                f"""
                Is there something important to memorize from the data? If so tell short and precise what it is. Only information that is useful for the Plan.
                Are there obsolete or redundat MEMORIES that need to be deleted? Keep only important ones.
                \n:{context}
                \n{task}
                """
            ),
            realization_context=(
                f"Memories: {str(self.main_memory.memories)}\n"
                f"{context}\nReasoning: {{imagination}}\n"
                "Use the tools to store and/or delete memories like told. If there is nothing to do use the noop tool."
            ),
            tools=ToolGroup.MEM,
            realisator_system="Respond by calling the given functions only!",
            imaginator_name="memo_imaginator",
            realisator_name="memo_realisator",
            allow_question_retry=False,
        )

        return imagination

    def _parse_decision(self, response: str, positive_markers=None) -> bool:
        markers = ["done", "yes", "y", "true"] if positive_markers is None else positive_markers
        normalized = response.strip().lower()
        return any(normalized.startswith(marker) for marker in markers)

    def _ask_yes_no(self, context: str, question: str) -> bool:
        """Use imaginator reasoning and QA tool calls to answer a binary question."""

        current.ANSWER_BUFFER = None

        imaginator = Provider.build("qa_imaginator", self.imaginator_model, memory=self.main_memory)
        reasoning = imaginator.call(
            f"{context}\nQuestion: {question}\nProvide a short reasoning before deciding."
        )

        qa_memory = Memory()
        qa_memory.append_message(
            Role.USER,
            f"{context}\nQuestion: {question}\nReasoning: {reasoning}\nUse yes() or no() tool calls only.",
        )

        qa_realisator = ToolProvider.build("qa_realisator", self.realisator_model, qa_memory)
        register_tools(qa_realisator, ToolGroup.QA)
        qa_realisator.invoke("Respond by calling yes() or no() to answer the question above.")

        return bool(current.ANSWER_BUFFER)

    def _format_plan_tree(self, node=None, prefix: str = "", active_node=None) -> str:
        node = node or self.main_memory.plan_root
        active_node = active_node or self.main_memory.plan_node

        marker = " (current focus)" if active_node and node.id == active_node.id else ""
        id_label = self._node_identifier(node)
        lines = [f"{prefix}- {id_label} {node.data}{marker}"]

        for child in node.children:
            lines.append(self._format_plan_tree(child, prefix + "  ", active_node))

        return "\n".join(lines)

    def _node_identifier(self, node) -> str:
        label = self._positional_label(node)
        if label:
            return f"[{node.id} | {label}]"
        return f"[{node.id}]"

    def _positional_label(self, node) -> str:
        if node is self.main_memory.plan_root:
            return ""

        path_indices = []
        current = node
        while current.parent:
            path_indices.append(current.parent.children.index(current))
            current = current.parent

        path_indices.reverse()
        label_parts = []
        token_generators = [self._numeric_token, self._upper_token, self._lower_token]

        for depth, index in enumerate(path_indices):
            token_fn = token_generators[depth % len(token_generators)]
            label_parts.append(token_fn(index))

        return "".join(label_parts)

    def _numeric_token(self, index: int) -> str:
        return str(index + 1)

    def _upper_token(self, index: int) -> str:
        return self._alpha_token(index, uppercase=True)

    def _lower_token(self, index: int) -> str:
        return self._alpha_token(index, uppercase=False)

    def _alpha_token(self, index: int, uppercase: bool) -> str:
        base = ord("A") if uppercase else ord("a")
        token = ""
        n = index
        while True:
            n, remainder = divmod(n, 26)
            token = chr(base + remainder) + token
            if n == 0:
                break
            n -= 1
        return token

    def _leaf_nodes(self, node=None) -> list:
        node = node or self.main_memory.plan_root

        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._leaf_nodes(child))

        return leaves

    def _choose_active_leaf(self) -> "PlanNode | None":
        leaves = self._leaf_nodes()
        if not leaves:
            return None

        current_focus = self.main_memory.plan_node if self.main_memory.plan_node in leaves else None

        if current_focus:
            should_keep = self._ask_yes_no(
                context=(
                    f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree(active_node=current_focus)}"
                ),
                question="Is the current focus still the best leaf task to pursue next?",
            )
            if should_keep:
                return current_focus

        chooser = Provider.build("leaf_selector", self.imaginator_model, memory=self.main_memory)
        leaf_listing = "\n".join([f"[{leaf.id}] {leaf.data}" for leaf in leaves])
        choice = chooser.call(
            f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree()}\n"
            f"Available leaf tasks:\n{leaf_listing}\n"
            "Select the best suited leaf task by replying with its id and a short justification.",
        )

        selected_leaf = None
        for leaf in leaves:
            if str(leaf.id) in choice:
                selected_leaf = leaf
                break

        if selected_leaf is None:
            selected_leaf = leaves[0]

        self.main_memory.plan_node = selected_leaf
        return selected_leaf

    def _high_level_decomposition(self):
        for _ in range(0, 3):
            current_plan = self._format_plan_tree()
            is_ready = self._ask_yes_no(
                context=(
                    f"Goal: {self.goal}\nCurrent plan tree:\n{current_plan}\n"
                    "Focus is a leaf task that can be executed directly."
                ),
                question="Is the current plan tree sufficiently decomposed to start execution?",
            )

            if is_ready:
                return

            self._imaginator_realisator_step(
                imagination_task=(
                    "Study the current plan tree and decide how to refine it."
                    " Suggest concise updates to decompose or prune nodes before execution."
                ),
                realization_context=(
                    f"Goal: {self.goal}\nPlan tree:\n{current_plan}\n"
                    "Use decompose_node(task_node_id, sub_nodes) to split tasks or delete_node(task_node_id, delete_children)"
                    " to remove or merge tasks."
                ),
                tools=ToolGroup.DECOMPOSE,
                realisator_system="Use only the provided plan editing tools to update the plan tree.",
                imaginator_name="decompose_imaginator",
                realisator_name="decompose_realisator",
                allow_question_retry=False,
            )

    def _generate_low_level_plan(self, active_node):
        self.main_memory.plan_steps.clear()

        self._imaginator_realisator_step(
            imagination_task=(
                f"Active target: [{active_node.id}] {active_node.data}."
                " Brainstorm concrete ideas on how to achieve this task."
                " Avoid repeating previously tried approaches."
            ),
            realization_context=(
                f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree(active_node=active_node)}\n"
                f"Completed items: {self.main_memory.completed_steps}\n"
                "Use add_step(text) for each distinct idea to try. Keep them concise and actionable."
            ),
            tools=ToolGroup.PLAN,
            realisator_system="Respond only with add_step() tool calls, one per idea.",
            imaginator_name="idea_imaginator",
            realisator_name="idea_realisator",
            allow_question_retry=False,
        )

    def _ensure_active_goal(self, active_node):
        goal_completed = self._ask_yes_no(
            context=(
                f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree(active_node=active_node)}"
            ),
            question="Is the focused leaf task completed?",
        )

        if goal_completed:
            self.main_memory.completed_steps.append(f"[{active_node.id}] {active_node.data}")
            if active_node.parent:
                self.main_memory.delete_plan_node(active_node.id)
            self.main_memory.plan_steps.clear()
            self.main_memory.plan_node = self.main_memory.plan_root
            return True

        return False

    def _evaluate_current_idea(self, active_node):
        if not self.main_memory.plan_steps:
            self._generate_low_level_plan(active_node)

        if not self.main_memory.plan_steps:
            return

        current_idea = self.main_memory.plan_steps[0]
        idea_viable = self._ask_yes_no(
            context=(
                f"Goal: {self.goal}\nActive task: [{active_node.id}] {active_node.data}\n"
                f"Current idea: {current_idea}"
            ),
            question="Does this idea still seem promising based on the latest perception and reflection?",
        )

        if idea_viable:
            return

        self.main_memory.completed_steps.append(self.main_memory.plan_steps.pop(0))
        if not self.main_memory.plan_steps:
            self._generate_low_level_plan(active_node)

    def act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        
        
        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But always tell the next action"

        if config.ACTIVE_CONFIG.agent.imaginator is config.ImaginatorType.OFF:
            self._direct_action_step(perception, prompt)
            return

        self._imaginator_realisator_step(
            imagination_task=prompt,
            realization_context=perception,
            tools=ToolGroup.ENV,
            realisator_system=(
                "Give exactly the toolcalls that arise from the planned action. Use correct object id. Toolcall order matters. "
                "If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a "
                "precice and short question."
            ),
            imaginator_name="action_imaginator",
            realisator_name="action_realisator",
            allow_question_retry=True,
        )

    def reflect(self, perception: str):
        action_messages = process_action_results()
        for role, msg, msg_type in action_messages:
            self.main_memory.append_message(role, msg, msg_type)

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            return

        mem = Memory()
        mem.append_message(Role.USER, perception)
        for role, msg, msg_type in action_messages:
            mem.append_message(role, msg, msg_type)
        reflector = Provider.build("reflector", self.imaginator_model, memory=mem)
        reflection = reflector.call("Reflect what effect the performed Actions had. Only what you can say for sure. (short)")
        self.main_memory.append_message(Role.USER, reflection, Type.REFLECT)

        if config.ACTIVE_CONFIG.agent.observe is config.ReflectType.MEMORIZE:
            self.memorize(reflection)

    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return

        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)



        if self.main_memory.plan is not None:
            if config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
                is_completed = self._ask_yes_no(
                    context=f"{self.main_memory.plan}",
                    question="Based on the current reflection of the last action, is the current plan fully completed?",
                )

                if is_completed:
                    self.main_memory.mark_completed()

                    if not self.main_memory.plan_steps:
                        should_finish = self._ask_yes_no(
                            context=f"Main goal: {self.goal}",
                            question="Is the main goal already achieved?",
                        )

                        if should_finish:
                            raise Exception("AGENT FINISHED")

                        return

                    current_step = self.main_memory.plan_steps[0]
                    self.main_memory.add_plan("PLAN: "+ current_step)
                    return

            if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.DECOMPOSE:
                should_plan = self._ask_yes_no(
                    context=(
                        f"Goal: {self.goal}\nCurrent plan: {self.main_memory.plan}\n"
                    ),
                    question="Based on the current goal, context, and the last action, is the current sub goal still possible?",
                )

                if should_plan:
                    return

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            plan = planner.invoke(
                "MAIN GOAL: " + self.goal,
                "What goals are there. Can one be broken down into sub goals. You must drop a subgoal immediately if the environment state proves that the subgoal is completed or impossible. Do not repeat subgoals that cannot change the environment anymore. Give structured listing. Tell about how to structuredly approach the next step.",
                append=False,
            )
            self.main_memory.add_plan(plan)
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
            self.make_structured_plan()
            self.completed_steps = []
            current_step = self.main_memory.plan_steps[0]
            self.main_memory.add_plan("PLAN: "+ current_step)
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
            self._high_level_decomposition()

            active_node = self._choose_active_leaf()
            if active_node is None:
                raise Exception("No available plan nodes to execute")

            if self._ensure_active_goal(active_node):
                active_node = self._choose_active_leaf()
                if active_node is None:
                    raise Exception("AGENT FINISHED")

            self._evaluate_current_idea(active_node)

            active_idea = self.main_memory.plan_steps[0] if self.main_memory.plan_steps else "No idea available"
            plan_overview = self._format_plan_tree(active_node=active_node)
            self.main_memory.add_plan(
                "FULL PLAN TREE:\n" + plan_overview +
                f"\nCurrent target: [{active_node.id}] {active_node.data}\n" +
                f"Current idea: {active_idea}"
            )
        else:
            raise NotImplementedError()
        
    #def is_ready(self):

    def make_structured_plan(self):
        #create plan (replan)

        #imaginate plan
        
        #get memory + goal without plan 
        plan_imaginator = Provider.build("plan_imaginator", self.imaginator_model, memory=self.main_memory)
        goal = self.goal

        prompt = f"MAIN GOAL: {goal} Create a strucutred plan to reach the goal."

        ##if self.main_memory.completed_steps:
        ##    prompt += "Old plan: " +  + ". Already completed subgoals: " + str(self.completed_steps)

        plan_str = plan_imaginator.call(prompt)

        mem = Memory()
        mem.append_message(Role.USER, plan_str)
        plan_realisator = ToolProvider.build("plan_realisator", self.realisator_model, mem)
        register_tools(plan_realisator, ToolGroup.PLAN)

        correction_note = ""
        for _ in range(0, 3):
            reply = plan_realisator.invoke(
                "Use toolcalls to add the steps specified in the plan for the user." + correction_note
            )

            has_error, errors = process_formal_errors(plan_realisator.memory, collect=True)
            if not has_error:
                break

            self.main_memory.plan_steps.clear()
            hint_texts = [e.get("hint") for e in errors if e.get("hint")]
            agent_msgs = [e.get("agent_message") for e in errors if e.get("agent_message")]
            combined = "; ".join(hint_texts or agent_msgs)
            if combined:
                correction_note = f" Retry with these corrections in mind: {combined}. Keep tool calls concise."
            else:
                correction_note = " Retry with strict, valid tool calls only."

        if not self.main_memory.plan_steps:
            raise Exception("no planned steps")


    def _imaginator_realisator_step(
        self,
        imagination_task: str,
        realization_context: str,
        tools: ToolGroup,
        realisator_system: str,
        imaginator_name: str = "imaginator",
        realisator_name: str = "realisator",
        allow_question_retry: bool = False,
        realisator_memory: Memory | None = None,
    ) -> str:
        imaginator = Provider.build(imaginator_name, self.imaginator_model, memory=self.main_memory)

        imagination = imaginator.call(imagination_task)

        realisator_mem = realisator_memory or Memory(realisator_system)
        realisator = ToolProvider.build(
            realisator_name,
            self.realisator_model,
            realisator_mem,
        )
        register_tools(realisator, tools)

        correction_suffix = ""
        final_has_error = False
        last_error_payloads: list[dict] = []

        for attempt in range(0, 3):
            # Avoid Python format treating braces from JSON perceptions as placeholders.
            # Only the explicit `{imagination}` token should be substituted.
            ctx = realization_context.replace("{imagination}", imagination) + correction_suffix
            reply = realisator.invoke(ctx)

            has_error, errors = process_formal_errors(realisator.memory, collect=True)
            final_has_error = has_error
            last_error_payloads = errors

            if not has_error:
                break

            hint_texts = [e.get("hint") for e in errors if e.get("hint")]
            agent_msgs = [e.get("agent_message") for e in errors if e.get("agent_message")]
            combined = "; ".join(hint_texts or agent_msgs)
            correction_suffix = (
                f" Retry with these corrections: {combined}. Keep tool calls minimal."
                if combined
                else " Retry using valid tool calls with explicit object IDs."
            )

            if not allow_question_retry or config.ACTIVE_CONFIG.agent.imaginator is not config.ImaginatorType.QUESTION:
                self.main_memory.append_message(
                    Role.USER,
                    "Last step was not specified well. Please provide more explicit instructions",
                )
                break

            if "Question" in reply:
                correction_suffix = imaginator.invoke(
                    reply + " Use explicit object IDs and absolute positions.",
                    imagination_task + " Provide an explicit and precise instruction.",
                    append=False,
                )
            else:
                break

        if final_has_error:
            hint_texts = [e.get("hint") for e in last_error_payloads if e.get("hint")]
            agent_msgs = [e.get("agent_message") for e in last_error_payloads if e.get("agent_message")]
            combined = "; ".join(hint_texts or agent_msgs)
            FormalError(
                "Action generation failed after multiple retries.",
                console_message=(
                    "Action generation failed after multiple retries. "
                    "Review the latest hints and adjust the prompt."
                ),
                hint=combined or None,
                context={
                    "imagination_task": imagination_task,
                    "realization_context": realization_context,
                    "attempts": attempt + 1,
                },
            )
            process_formal_errors(self.main_memory)

        return imagination

    def _direct_action_step(self, context: str, task: str):
        realisator = ToolProvider.build(
            "action_realisator",
            self.realisator_model,
            Memory(
                "Use concise, executable actions. Prefer tool calls when possible; otherwise respond with a short action descriptio"
                "n."
            ),
        )
        register_tools(realisator, ToolGroup.ENV)
        realisator.invoke(context + ". " + task)
