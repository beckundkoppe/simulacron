from agent.helper import process_action_results, process_formal_errors
from agent.toolpool import ToolGroup, register_tools
import config
import current
from enviroment.entity import Entity
from llm.cache import Cache
from llm.memory.memory import Memory, Role, SummarizingMemory
from llm.model import Model
from llm.provider import Provider
from llm.toolprovider import ToolProvider

class Agent:
    def __init__(self, goal: str, entity: Entity, imaginator_model: Model, realisator_model: Model, extra_model: Model):
        self.goal = goal
        self.entity = entity
        self.imaginator_model = imaginator_model
        self.realisator_model = realisator_model
        self.extra_model = extra_model

        self.main_memory = SummarizingMemory(model=extra_model)

        self.current_plan = "none yet."

    def update(self, observation: str):
        current.ENTITY = self.entity

        self.plan()
        self.observe(observation)
        self.act(observation)
        self.reflect(observation)

        current.ENTITY = None

    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return
        
        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)
        self.current_plan = planner.invoke("GOAL: " + self.goal,  "What are the steps to reach the goal? What ist the plan?", append=False)

    def observe(self, observation: str):
        if config.ACTIVE_CONFIG.agent.observe is config.PlanType.OFF:
            return
        
        self.memorize()

    def memorize(self):
        pass

    def act(self, observation: str):
        if config.ACTIVE_CONFIG.agent.imaginator is config.ImaginatorType.OFF:
            self.provider = Provider.build("agent", self.realisator_model, memory=self.main_memory)
            raise NotImplementedError
        else:
            self._imgagination_realisation_step(observation, " Give best next action to perform (short answer)", ToolGroup.ENV)

    def reflect(self, observation: str):
        process_action_results(self.main_memory)

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            return
        
        self.learn()

    def learn(self):
        pass

    def _imgagination_realisation_step(self, context, task: str, tools: ToolGroup):
        imaginator = Provider.build("imaginator", self.imaginator_model, memory=self.main_memory)
        imaginator.memory.add_message(Role.SYSTEM, "You are an advanced AI.")
        imagination = imaginator.invoke(context, task)
        self.main_memory.save("main_memory.txt")

        realisator = ToolProvider.build("realisator", self.realisator_model, Memory())
        register_tools(realisator, ToolGroup.ENV)

        correction = None

        for i in range(0, 3):
            if correction is None:
                assert i == 0, "Imaginator failed"
                ctx = "PLAN: " + imagination
            else:
                ctx = correction

            retry = ""
            for i in range(0, 3):
                #imaginator.memory.add_message(Role.SYSTEM, "Give exactly the toolcalls that arise from the planned action. Use correct object id. Toolcall order matters. If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a precice and short question.")
                #reply = realisator.invoke(ctx, retry + "Give the toolcalls")
                reply = realisator.invoke("What actions are available?")

                if process_formal_errors(realisator.memory):
                    #helper = Provider.build("helper", self.imaginator_model, memory=self.main_memory)
                    #why = helper.invoke("Precise instruction how to prevent the error. (short)", override=realisator.memory, append=False)
                    retry = "Retry with corrected version: "
                else:
                    break;

            correction = None

            if config.ACTIVE_CONFIG.agent.imaginator is not config.ImaginatorType.QUESTION:
                self.main_memory.add_message(Role.USER, "Last action was not specified well. Please provide more explicit instructions")
                self.main_memory.save("main_memory.txt")
                break

            if "Question" in reply:
                correction = imaginator.invoke(reply + "Use explicit object IDs and absolute positions", task + " Provide an explicit and precice instruction", override=self.main_memory)
                self.main_memory.save("main_memory.txt")
            else:
                break