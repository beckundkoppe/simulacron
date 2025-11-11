import config
from enviroment.entity import Entity
from llm.cache import Cache
from llm.memory.memory import Role, SummarizingMemory
from llm.model import Model
from llm.toolprovider import ToolProvider

class Agent:
    def __init__(self, goal: str, entity: Entity, cache: Cache, imaginator_model: Model, realisator_model: Model, extra_model: Model):
        self.goal = goal
        self.entity = entity
        self.cache = cache
        self.extra_model = extra_model

        if(config.ACTIVE_CONFIG.agent.imaginator):
            self.imaginator: ToolProvider = ToolProvider.build(cache.get(imaginator_model))
            self.realisator: ToolProvider = ToolProvider.build(cache.get(imaginator_model))
        else:
            config.ACTIVE_CONFIG.agent.imaginator:


    def update(self, observation: str):
        self.plan()
        self.observe()
        self.act()
        self.reflect()

    def plan(self):
        if config.ACTIVE_CONFIG.agent.plan is config.PlanType.OFF:
            return

    def observe(self):
        if config.ACTIVE_CONFIG.agent.observe is config.PlanType.OFF:
            return

    def act(self):
        pass

    def reflect(self):
        if config.ACTIVE_CONFIG.agent.reflect is config.PlanType.OFF:
            return
        
        self.learn()

    def learn(self):
        pass

    def learn(self):
        pass
    