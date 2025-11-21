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

        self.observe(perception)
        self.main_memory.save()

        self.plan()
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
        pass
    
    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return
        
        self.plan()

    def act(self, perception: str):
        prompt = "Give best next action to perform (short answer)."

        
        
        if config.ACTIVE_CONFIG.agent.plan is not config.PlanType.OFF:
            prompt += " Stick closely to the next step in the plan. But allways tell the next action"

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
        self.reflection = reflector.call("Reflect what effect the performed Actions had. Only what you can say for sure. (short)")
        self.main_memory.append_message(Role.USER, self.reflection, Type.REFLECT)
        
        self.learn()

    def learn(self):
        pass

    def plan(self):
        planner = Provider.build("planner", self.imaginator_model, memory=self.main_memory)

        if self.main_memory.plan is not None:
            if config.ACTIVE_CONFIG.agent.plan is config.PlanType.STRUCTURED:
                is_completed: str = planner.invoke(
                    "PLAN:" + self.main_memory.plan + " REFLECTION: "+self.reflection,
                    "Based on the current reflection of the last action. Is the current PLAN completed? Answer with 'yes' or 'no' (nothing more).",
                    override=Memory()
                )

                if "yes" in is_completed.lower():
                    self.main_memory.mark_completed()
                    print(self.main_memory.plan_steps)

                    if not self.main_memory.plan_steps:
                        should_plan: str = planner.call(
                            "Based on the current goal, context, and the last action: "
                            "Have we completed the main Goal?"
                            "Answer with 'yes' or 'no' (nothing more)."
                        )

                        if "yes" in should_plan.lower():
                            raise Exception("AGENT FINISHED")
                        
                        return
                        
                    current_step = self.main_memory.plan_steps[0]
                    self.main_memory.add_plan("PLAN: "+ current_step)
                    return    

            should_plan: str = planner.call(
                "Based on the current goal, context, and the last action: "
                "Is the current sub goal still possible?. "
                "Answer with 'yes' or 'no' (nothing more)."
            )

            if "yes" in should_plan.lower():
                return
            
            #conclusion: str = planner.call(
            #    "Based on the current goal, context, and the last action: "
            #    "Why is the current plan outdated?"
            #)
                
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.FREE:
            plan = planner.invoke("MAIN GOAL: " + self.goal, "What goals are there. Can one be broken down into sub goals. You must drop a subgoal immediately if the environment state proves that the subgoal is completed or impossible. Do not repeat subgoals that cannot change the environment anymore. Give structured listing. Tell about how to structuredly approach the next step.", append=False)                
            self.main_memory.add_plan(plan)
        elif config.ACTIVE_CONFIG.agent.plan is config.PlanType.STRUCTURED:
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

    
        plan_realisator = ToolProvider.build("plan_realisator", self.realisator_model, Memory("Use toolcalls to add the following steps that arise with the given plan"))
        register_tools(plan_realisator, ToolGroup.PLAN)

        for i in range(0, 3):
            reply = plan_realisator.invoke(plan_str)

            if process_formal_errors(None):
                self.main_memory.plan_steps.clear()
            else:
                break
        
        print(self.main_memory.plan_steps)
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

            realisator = ToolProvider.build("realisator", self.realisator_model, Memory("Give exactly the toolcalls that arise from the planned action. Use correct object id. Toolcall order matters. If there are implicit references to vague to be realised with the available tools answer with 'Question:' and a precice and short question."))
            register_tools(realisator, tools)

            correction = None

            for i in range(0, 3):
                if correction is None:
                    assert i == 0, "Imaginator failed"
                    ctx = context + ". What to do next: " + imagination
                else:
                    ctx = context + ". What to do next: " + imagination + correction

                for i in range(0, 3):
                    reply = realisator.invoke(ctx)
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