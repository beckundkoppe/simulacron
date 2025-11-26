from typing import Tuple
from agent.helper import process_action_results, process_formal_errors
from agent.plan import FreePlan, PlanNode, StepPlan, TreePlan
from agent.toolpool import ToolGroup, register_tools
import config
import current
from enviroment.entity import Entity
from enviroment.resultbuffer import FormalError
from llm.memory.memory import Memory, Role, Type
from llm.memory.supermem import SuperMemory
from llm.model import Model
from llm.provider import Provider
from llm.toolprovider import ToolProvider
from util.console import Color, bullet, pretty, title

class Agent:
    def __init__(self, goal: str, entity: Entity, imaginator_model: Model, realisator_model: Model):
        self.entity = entity
        self.imaginator_model = imaginator_model
        self.realisator_model = realisator_model

        self.goal = goal
        self.main_memory = SuperMemory(goal=goal, path="main_memory.txt")

        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            self._replan("Initial main goal is not yet reached")

    def update(self, perception: str):
        current.AGENT = self
        current.ENTITY = self.entity

        self.plan = self._plan() # should set self.plan with a full Plan
        self.main_memory.save()

        observation = self._observe(perception) # returns aufbereitete observation mit den für die aufgabe relevanten inhalte
        self.main_memory.save()

        ideas = self._trial() # generiert List[str] mit ideen um das ziel im focused plannode zu erreichen
        self.main_memory.save()

        self._act(perception) # führt beste idee aus
        self.main_memory.save()

        self._reflect(perception) # stellt fest, was das ergebniss der aktion ist. ziel im focused plannode schon erreicht? dann in planung, sonst Frage: sind wir noch auf einem guten weg? ja -> weiter; nein -> nächste idee
        self.main_memory.save()

        current.AGENT = None
        current.ENTITY = None

    def _observe(self, perception: str):
        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.OFF:
            self.main_memory.append_message(Role.USER, perception, Type.PERCEPTION)
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
        observation = observer.call(f"Perception: {perception}\nWhat do you observe? What is relevant for your {active_plan}? (short)")

        if config.ACTIVE_CONFIG.agent.observe is config.ObserveType.OFF:
            self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)
        else:
            pass
            #self._memorize(observation, Type.OBSERVATION)

        #if config.ACTIVE_CONFIG.agent.observe is not config.ObserveType.ON:
            #self._memorize(observation, "Do you now this room already?")

    def _trial(self):
        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.DECOMPOSE:
            return None

        if not isinstance(self.plan, TreePlan):
            return None

        active_node = self.plan.focus
        if active_node is None or active_node.done:
            return None

        if self._ensure_active_goal(active_node):
            return None

        self._evaluate_current_idea(active_node)

        active_idea = self.plan.plan_steps[0] if self.plan.plan_steps else "No idea available"
        plan_overview = self._format_plan_tree(active_node=active_node)
        self.main_memory.add_plan(
            "FULL PLAN TREE:\n" + plan_overview +
            f"\nCurrent target: [{active_node.id}] {active_node.data}\n" +
            f"Current idea: {active_idea}"
        )

        return active_node
    
    def _act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But always tell the next action"

        if config.ACTIVE_CONFIG.agent.action is config.ActionType.DIRECT:
            realisator = ToolProvider.build("actor", self.realisator_model, memory=self.main_memory)
            register_tools(realisator, ToolGroup.ENV)
            realisator.call(perception + ". " + prompt + "Use concise, executable actions. Your answer must only consist of toolcalls.")
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

    def _reflect(self):
        results = ""
        for role, msg, msg_type in process_action_results():
            results += msg + "\n"

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            self.main_memory.append(Type.FEEDBACK, Role.USER, results)
            return

        reflector = Provider.build("reflector", self.imaginator_model, memory=self.main_memory)
        reflection = reflector.call("Result: {result}\nReflect what effect the performed Actions had. Only what you can say for sure. (short)")

        self.main_memory.append_message(Role.USER, reflection, Type.REFLECT)

        #if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.ON:
        #    self.main_memory.append_message(Role.USER, reflection, Type.REFLECT)
        #else:
        ##    self._memorize(reflection)

    def _format_plan_tree(self, node=None, prefix: str = "", active_node=None) -> str:
        if isinstance(self.plan, TreePlan):
            return self.plan.format_plan_tree(node=node, prefix=prefix, active_node=active_node)
        return ""

    def _leaf_nodes(self, node=None) -> list[PlanNode]:
        if isinstance(self.plan, TreePlan):
            return self.plan.leaf_nodes(node=node)
        return []

    def _choose_active_leaf(self) -> "PlanNode | None":
        leaves = self._leaf_nodes()
        if not leaves:
            return None

        current_focus = self.plan.focus if isinstance(self.plan, TreePlan) and self.plan.focus in leaves else None

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

        if isinstance(self.plan, TreePlan):
            self.plan.focus = selected_leaf
        return selected_leaf

    def _decompose_tree_plan(self):
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
                    f"Goal: {self.goal}\nCurrent plan tree:\n{current_plan}\n"
                    "Study the current plan tree and decide how to refine it."
                    " Suggest concise updates to decompose or prune nodes before execution."
                ),
                realization_context=(
                    f"Goal: {self.goal}\nPlan tree:\n{current_plan}\n"
                    "Use decompose_node(task_node_id, sub_nodes) to split tasks, delete_node(task_node_id, delete_children)"
                    " to remove or merge tasks, mark_done(task_node_id) to label completed nodes, and"
                    " mark_focued(task_node_id) to set the next focus."
                ),
                tools=ToolGroup.DECOMPOSE,
                realisator_prompt="Use only the provided plan editing tools to update the plan tree.",
                name="decompose",
                allow_question_retry=False,
            )

    def _generate_low_level_plan(self, active_node):
        if not isinstance(self.plan, TreePlan):
            return

        self.plan.plan_steps.clear()

        self._imaginator_realisator_step(
            imagination_task=(
                f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree(active_node=active_node)}\n"
                f"Active target: [{active_node.id}] {active_node.data}."
                " Brainstorm concrete ideas on how to achieve this task."
                " Avoid repeating previously tried approaches."
                " If the naive approach hasent been tried yet, try it."
            ),
            realization_context=(
                f"Goal: {self.goal}\nPlan tree:\n{self._format_plan_tree(active_node=active_node)}\n"
                f"Completed items: {self.plan.completed_steps}\n"
                "Use add_step(text) for each distinct idea to try. Keep them concise and actionable."
            ),
            tools=ToolGroup.PLAN,
            realisator_prompt="Respond only with add_step() tool calls, one per idea.",
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

        if goal_completed and isinstance(self.plan, TreePlan):
            self.plan.mark_node_done(active_node.id)
            self.plan.plan_steps.clear()
            self.plan.focus = self.plan.root
            return True

        return False

    def _evaluate_current_idea(self, active_node):
        if not isinstance(self.plan, TreePlan):
            return

        if not self.plan.plan_steps:
            self._generate_low_level_plan(active_node)

        if not self.plan.plan_steps:
            return

        current_idea = self.plan.plan_steps[0]
        idea_viable = self._ask_yes_no(
            context=(
                f"Goal: {self.goal}\nActive task: [{active_node.id}] {active_node.data}\n"
                f"Current idea: {current_idea}"
            ),
            question="Does this idea still seem promising based on the latest perception and reflection?",
        )

        if idea_viable:
            return

        self.plan.completed_steps.append(self.plan.plan_steps.pop(0))
        if not self.plan.plan_steps:
            self._generate_low_level_plan(active_node)

    def _plan(self):
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

            if ans2 is True:
                self._replan(rationale2)
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
            self.finished = True
            return
        else:
            self._replan("Initial main goal is not yet reached")

    def _replan(self, reason):
        print(f"Replanning because: {reason}")

        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            self.plan = self.plan_free()
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
            self.plan = self.plan_steps()
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.DECOMPOSE:
            self.plan = self.plan_tree()
        else:
            raise Exception()

        self.main_memory.add_plan(self.plan)
                

    def _plan_free(self, replan_reason=None) -> FreePlan:
        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)
        plan_text = planner.call(
            "MAIN GOAL: "
            + self.goal
            + "\n"
            + "What goals are there. Can one be broken down into sub goals. You must drop a subgoal immediately if the environment state proves that the subgoal is completed or impossible. Do not repeat subgoals that cannot change the environment anymore. Give structured listing. Tell about how to structuredly approach the next step."
        )

        return FreePlan(goal=self.goal, description=plan_text)

    def _plan_steps(self, replan_reason=None) -> StepPlan:
        step_plan = StepPlan(goal=self.goal)
        self.plan = step_plan

        self._imaginator_realisator_step(
            imagination_task=(f"Main goal: {self.goal}\nReplanning because {replan_reason}\nCreate a new strucutred plan to reach the goal."),
            realization_context=(f"Plan: {{imagination}}\nUse toolcalls to add the steps specified in the plan for the user."),
            tools=ToolGroup.PLAN,
            name="plan"
        )

        return step_plan
    def _plan_tree(self, replan_reason=None) -> TreePlan:
        self.plan = TreePlan.new(self.goal)
        self.main_memory.add_plan(self.plan)

        self._decompose_tree_plan()

        active_node = self._choose_active_leaf()
        if active_node is None:
            raise Exception("No available plan nodes to execute")

        if isinstance(self.plan, TreePlan):
            self.plan.focus = active_node
            plan_overview = self._format_plan_tree(active_node=active_node)
            self.main_memory.add_plan(
                "FULL PLAN TREE:\n" + plan_overview +
                f"\nCurrent target: [{active_node.id}] {active_node.data}"
            )

        return self.plan

    def _imaginator_realisator_step(
        self,
        imagination_task: str,
        realization_context: str,
        tools: ToolGroup,
        name: str,
    ) -> None:
        imaginator = Provider.build(name + "_imaginator", self.imaginator_model, memory=self.main_memory)

        imagination = imaginator.call(imagination_task)

        realisator = ToolProvider.build(
            name + "_realisator",
            self.realisator_model,
            Memory(realization_context),
        )
        register_tools(realisator, tools)

        correction_suffix = ""
        final_has_error = False
        last_error_payloads: list[dict] = []

        for attempt in range(0, 3):
            ctx = realization_context.replace("{imagination}", imagination) + correction_suffix
            reply = realisator.invoke(ctx)

            has_error, errors = process_formal_errors(realisator.memory, collect=True)
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