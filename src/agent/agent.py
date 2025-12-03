import time
from typing import Tuple
from agent.helper import process_action_results, process_formal_errors
from agent.plan import FreePlan, PlanNode, StepPlan, TreePlan
from agent.toolpool import ToolGroup, register_tools
import config
import current
import debug
from enviroment.entity import Entity
from enviroment.resultbuffer import FormalError
from llm.memory.memory import Memory, Role, Type
from llm.memory.supermem import SuperMemory
from llm.model import Model
from llm.provider import Provider
from llm.toolprovider import ToolProvider
from util.console import Color, banner, bullet, pretty

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
                "The human agent can move, take and drop objects, and interact with objects (open, close, lock, unlock, go through or look through doors). Always choose the minimal action sequence that achieves the requsted goal and nothing else.",
            Type.GOAL
        )

        self.had_action = False
        self.triggered_replan = None
        self.finished = False

        if config.ACTIVE_CONFIG.agents.plan is not config.PlanType.OFF:
            self.triggered_replan = "Main goal is not yet reached. Create an initial plan"

        debug.print_to_file(self.main_memory.get_history())

    def _add_time(self, field: str, delta: float) -> None:
        result = getattr(current, "RESULT", None)
        if result is None:
            return
        setattr(result, field, getattr(result, field) + delta)

    def _create_main_memory(self, goal: str) -> Memory:
        mem_type = getattr(config.ACTIVE_CONFIG.agents, "memory_type", None)

        if mem_type is config.MemoryType.SUPER:
            return SuperMemory(goal=goal, path="main_memory.txt")

        return Memory(goal=goal, path="main_memory.txt")

    def update(self, perception: str):
        current.AGENT = self
        current.ENTITY = self.entity

        if not self.triggered_replan and self.had_action:
            t0 = time.time()
            self._reflect(perception) # stellt fest, was das ergebniss der aktion ist. ziel im focused plannode schon erreicht? dann in planung, sonst Frage: sind wir noch auf einem guten weg? ja -> weiter; nein -> nächste idee
            self._add_time("reflect_time_s", time.time() - t0)
            self.main_memory.save()
            self.had_action = False

        if self.triggered_replan or config.ACTIVE_CONFIG.agents.trial is config.TrialType.OFF:
            t0 = time.time()
            self._plan() # should set self.plan with a full Plan
            self._add_time("plan_time_s", time.time() - t0)
            self.main_memory.set_plan(self.plan)
            self.main_memory.save()
        
        if config.ACTIVE_CONFIG.agents.plan is not config.PlanType.OFF:
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

        if config.ACTIVE_CONFIG.agents.trial is not config.TrialType.OFF:
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
        if config.ACTIVE_CONFIG.agents.observe is config.ObserveType.OFF:
            self.main_memory.append_message(Role.USER, perception, Type.OBSERVATION)
            return

        if config.ACTIVE_CONFIG.agents.plan is config.PlanType.OFF:
            active_plan = "goal"
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.FREE:
            active_plan = "current plan step"
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.STEP:
            active_plan = "focused plan step"
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.DECOMPOSE:
            active_plan = "focused plan node"
        else:
            raise Exception()
        
        observer = Provider.build("observer", self.imaginator_model, memory=self.main_memory)
        start = time.time()
        observation = observer.call(f"Perception: {perception}\nWhat do you observe? Make sure to verify facts. Respect facts, not assumptions. What is relevant for your {active_plan}? Only tell about new discouveries. (short)")
        self._add_time("img_time_s", time.time() - start)

        if config.ACTIVE_CONFIG.agents.observe is config.ObserveType.ON:
            self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)
        else:
            pass
            #self._memorize(observation, Type.OBSERVATION)

    def _trial(self):
        if config.ACTIVE_CONFIG.agents.plan is config.PlanType.OFF:
            return
        
        if config.ACTIVE_CONFIG.agents.trial is config.TrialType.OFF:
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
            if config.ACTIVE_CONFIG.agents.plan is config.PlanType.FREE:
                plan = "current plan"
            elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.STEP:
                plan = "focused plan step"
            elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.DECOMPOSE:
                plan = "focused plan node"
            else:
                raise Exception()
            self.triggered_replan = f"failed to get {plan} done"
        else:
            print(f"Current trial {self.plan.get_trial().current_step()}")
    
    def _act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        if config.ACTIVE_CONFIG.agents.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But always tell the next action"

        if config.ACTIVE_CONFIG.agents.action is config.ActionType.DIRECT:
            realisator = ToolProvider.build("actor", self.realisator_model, memory=self.main_memory)
            register_tools(realisator, ToolGroup.ENV)
            start = time.time()
            realisator.invoke(perception + ". " + prompt + "Use concise, executable actions. Your answer must only consist of toolcalls.")
            self._add_time("real_time_s", time.time() - start)
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

        if config.ACTIVE_CONFIG.agents.reflect is config.ReflectType.OFF:
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

        if config.ACTIVE_CONFIG.agents.plan is config.PlanType.OFF:
            return

        if config.ACTIVE_CONFIG.agents.plan is config.PlanType.FREE:
            plan = "current plan"
            active_plan_completed = f"{plan} completed and the goal reached"
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.STEP:
            plan = "focused plan step"
            active_plan_completed = f"{plan} completed"
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.DECOMPOSE:
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

        if config.ACTIVE_CONFIG.agents.plan is config.PlanType.FREE:
            self.plan = self._plan_free(reason)
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.STEP:
            self.plan = self._plan_steps(reason)
        elif config.ACTIVE_CONFIG.agents.plan is config.PlanType.DECOMPOSE:
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
    ) -> None:
        imaginator = Provider.build(name + "_imaginator", self.imaginator_model, memory=self.main_memory)

        start_imagination = time.time()
        imagination = imaginator.call(imagination_task)
        self._add_time("img_time_s", time.time() - start_imagination)

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
            start_real = time.time()
            reply = realisator.invoke(ctx)
            self._add_time("real_time_s", time.time() - start_real)

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

            if config.ACTIVE_CONFIG.agents.action is config.ActionType.IMG_RETRY:
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
