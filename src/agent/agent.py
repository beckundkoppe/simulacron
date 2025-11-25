from agent.helper import process_action_results, process_formal_errors
from agent.toolpool import ToolGroup, register_tools
import config
import current
from enviroment.entity import Entity
from llm.memory.memory import Memory, Role, Type
from llm.memory.supermem import SuperMemory
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

        self.reflection = ""

    def update(self, perception: str):
        current.AGENT = self
        current.ENTITY = self.entity

        self.plan()
        self.main_memory.save()

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

        self.memorize()

    def memorize(self):
        if config.ACTIVE_CONFIG.agent.observe is not config.ObserveType.MEMORIZE:
            return

        last_perception = None
        for msg_type, _, msg in reversed(self.main_memory._history):
            if msg_type is Type.PERCEPTION:
                last_perception = msg
                break

        if last_perception:
            self.main_memory.append_message(
                Role.USER,
                f"Memorized snapshot: {last_perception}",
                Type.SUMMARY,
            )

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

    def act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        
        
        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But always tell the next action"

        if config.ACTIVE_CONFIG.agent.imaginator is config.ImaginatorType.OFF:
            self._direct_action_step(perception, prompt)
            return

        self._imgagination_realisation_step(perception, prompt, ToolGroup.ENV)

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
        self.reflection = reflector.call("Reflect what effect the performed Actions had. Only what you can say for sure. (short)")
        self.main_memory.append_message(Role.USER, self.reflection, Type.REFLECT)

        self.learn()

    def learn(self):
        if config.ACTIVE_CONFIG.agent.reflect is not config.ReflectType.MEMORIZE:
            return

        if not self.reflection:
            return

        self.main_memory.append_message(
            Role.USER,
            f"Lesson learned: {self.reflection}",
            Type.FEEDBACK,
        )

    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return

        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)



        if self.main_memory.plan is not None:
            if config.ACTIVE_CONFIG.agent.plan is config.PlanType.STEP:
                is_completed = self._ask_yes_no(
                    context=f"PLAN: {self.main_memory.plan}\nREFLECTION: {self.reflection}",
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

            should_plan = self._ask_yes_no(
                context=(
                    f"Goal: {self.goal}\nCurrent plan: {self.main_memory.plan}\nReflection: {self.reflection}\n"
                    f"Perception: {perception}"
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

        
        #activate step
        #(actionloop)
        #reflect: is step completed
        #   yes: next step
        #   no: retry OR replan 

    def _imgagination_realisation_step(self, context, task: str, tools: ToolGroup):
        imaginator = Provider.build("imaginator", self.imaginator_model, memory=self.main_memory)

        imagination = imaginator.call(task)

        realisator = ToolProvider.build(
            "realisator",
            self.realisator_model,
            Memory(
                "Give exactly the toolcalls that arise from the planned action. Use correct object id. Toolcall order matters. If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a precice and short question."
            ),
        )
        register_tools(realisator, tools)

        correction_suffix = ""

        for attempt in range(0, 3):
            ctx = context + ". What to do next: " + imagination + correction_suffix
            reply = realisator.invoke(ctx)

            has_error, errors = process_formal_errors(realisator.memory, collect=True)
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

            if config.ACTIVE_CONFIG.agent.imaginator is not config.ImaginatorType.QUESTION:
                self.main_memory.append_message(
                    Role.USER,
                    "Last action was not specified well. Please provide more explicit instructions",
                )
                break

            if "Question" in reply:
                correction_suffix = imaginator.invoke(
                    reply + " Use explicit object IDs and absolute positions.",
                    task + " Provide an explicit and precise instruction.",
                    append=False,
                )
            else:
                break

    def _direct_action_step(self, context: str, task: str):
        realisator = ToolProvider.build(
            "realisator",
            self.realisator_model,
            Memory(
                "Use concise, executable actions. Prefer tool calls when possible; otherwise respond with a short action descriptio"
                "n."
            ),
        )
        register_tools(realisator, ToolGroup.ENV)
        realisator.invoke(context + ". " + task)
