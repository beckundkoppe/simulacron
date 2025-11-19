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

    def update(self, observation: str):
        current.ENTITY = self.entity

        self.plan()
        self.main_memory.save()

        self.observe(observation)
        self.main_memory.save()

        self.act(observation)
        self.main_memory.save()

        self.reflect(observation)
        self.main_memory.save()

        current.ENTITY = None

    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return
        
        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)
        plan = planner.invoke("MAIN GOAL: " + self.goal, "What are the steps to reach the goal? Make a short plan or update the existing one", append=False)
        self.main_memory.add_plan(plan)

    def observe(self, observation: str):
        self.main_memory.append_message(Role.USER, observation, Type.OBSERVATION)

        if config.ACTIVE_CONFIG.agent.observe is config.PlanType.OFF:
            return
        
        mem = Memory()
        mem.append_message(Role.USER, observation)
        observer = Provider.build("observer", self.imaginator_model, memory=mem)
        active_observation = observer.call("What do you observe? What is interesting? (short)")
        self.main_memory.append_message(Role.USER, active_observation, Type.ACTIVE_OBSERVATION)
        
        self.memorize()

    def memorize(self):
        pass

    def act(self, observation: str):
        if config.ACTIVE_CONFIG.agent.imaginator is config.ImaginatorType.OFF:
            self.provider = Provider.build("agent", self.realisator_model, memory=self.main_memory)
            raise NotImplementedError
        else:
            self._imgagination_realisation_step(observation, "Give best next action to perform (short answer)", ToolGroup.ENV)

    def reflect(self, observation: str):
        result = process_action_results()
        self.main_memory.append(result)

        if config.ACTIVE_CONFIG.agent.reflect is config.ReflectType.OFF:
            return
        
        mem = Memory()
        mem.append_message(Role.USER, observation)
        mem.append(result)
        reflector = Provider.build("reflector", self.imaginator_model, memory=mem)
        reflection = reflector.call("Reflect what effect the performed Actions had.")
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