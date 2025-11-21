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

class Agent:
    def __init__(self, goal: str, entity: Entity, imaginator_model: Model, realisator_model: Model):
        self.goal = goal
        self.entity = entity
        self.imaginator_model = imaginator_model
        self.realisator_model = realisator_model

        self.goal = goal
        self.main_memory = SuperMemory(goal=goal, path="main_memory.txt")

    def update(self, perception: str):
        current.ENTITY = self.entity

        self.observe(perception)
        self.main_memory.save()

        self.plan()
        self.main_memory.save()

        self.act(perception)
        self.main_memory.save()

        self.reflect(perception)
        self.main_memory.save()

        current.ENTITY = None

    def observe(self, observation: str):
        self.main_memory.append_message(Role.USER, observation, Type.PERCEPTION)

        if config.ACTIVE_CONFIG.agent.observe is config.PlanType.OFF:
            return
        
        mem = Memory()
        mem.append_message(Role.USER, observation)
        observer = Provider.build("observer", self.imaginator_model, memory=mem)
        observation = observer.call("What do you observe? What is interesting? (short)")
        self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)
        
        self.memorize()

    def memorize(self):
        pass
    
    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return
        
        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)

        should_plan: str = planner.call(
            "Based on the current goal, context, and the last action: "
            "Do we need to replan? Only if really neccesarry. "
            "Answer with 'yes' or 'no' (nothing more)."
        )

        if "no" in should_plan.lower():
            return

        #prompt = """"You are the planning module of an embodied agent. Your task is to analyze the main goal and the current memory of the agent.
        #Produce a clean, updated plan consisting of concrete sub-goals in logical order.
#
        #Requirements:
        #1. Break the main goal into a minimal set of precise sub-goals.
        #2. Compare these sub-goals with the memory of past actions.
        #3. Remove all sub-goals that are already completed.
        #4. Keep only steps that are still required to reach the goal.
        #5. Output the plan as a simple ordered list.
        #6. Be concise and avoid explanations.
        #"""

        #plan = planner.invoke(prompt, "MAIN GOAL: " + self.goal, append=False)
        plan = planner.invoke("MAIN GOAL: " + self.goal, "What Goals are there. Can one be broken down into sub goals. Remove completed Goals. Give structured listing. Tell about how to structuredly approach the next step.", append=False)
        self.main_memory.add_plan(plan)

    def act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan."

        if config.ACTIVE_CONFIG.agent.imaginator is config.ImaginatorType.OFF:
            self.provider = Provider.build("agent", self.realisator_model, memory=self.main_memory)
            raise NotImplementedError
        else:
            self._imgagination_realisation_step(perception, prompt, ToolGroup.ENV)

    def reflect(self, perception: str):
        result = process_action_results()
        self.main_memory.append(result)

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            return
        
        mem = Memory()
        mem.append_message(Role.USER, perception)
        mem.append(result)
        reflector = Provider.build("reflector", self.imaginator_model, memory=mem)
        reflection = reflector.call("Reflect what effect the performed Actions had. Only what you can say for sure. (short)")
        self.main_memory.append_message(Role.USER, reflection, Type.REFLECT)
        
        self.learn()

    def learn(self):
        pass

    def _imgagination_realisation_step(self, context, task: str, tools: ToolGroup):
        imaginator = Provider.build("imaginator", self.imaginator_model, memory=self.main_memory)

        imagination = imaginator.call(task)

        realisator = ToolProvider.build("realisator", self.realisator_model, Memory("Give exactly the toolcalls that arise from the planned action. Use correct object id. Toolcall order matters. If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a precice and short question."))
        register_tools(realisator, tools)

        correction = None

        for i in range(0, 3):
            if correction is None:
                assert i == 0, "Imaginator failed"
                ctx = context + ". What to do next: " + imagination
            else:
                ctx = context + ". What to do next: " + imagination + correction

            retry = ""
            for i in range(0, 3):
                reply = realisator.invoke(ctx, retry)
                #reply = realisator.invoke("What actions are available?")

                if process_formal_errors(realisator.memory):
                    #helper = Provider.build("helper", self.imaginator_model, memory=self.main_memory)
                    #why = helper.invoke("Precise instruction how to prevent the error. (short)", override=realisator.memory, append=False)
                    #retry = "Retry with corrected version: "
                    pass
                else:
                    break

            correction = None

            if config.ACTIVE_CONFIG.agent.imaginator is not config.ImaginatorType.QUESTION:
                self.main_memory.append_message(Role.USER, "Last action was not specified well. Please provide more explicit instructions")
                break

            if "Question" in reply:
                correction = imaginator.invoke(reply + "Use explicit object IDs and absolute positions", task + " Provide an explicit and precice instruction", append=False)
            else:
                break