import copy
from typing import List, Tuple
import config
from llm.memory.memory import Memory, Role, Type


class SuperMemory(Memory):
    def __init__(self, goal, path) -> None:
        super().__init__(goal, path)

        #goal
        #_history
        self._learnings: List[str] = []
        self._plans: List[str] = []
        self._memories: List[str] = []
        self._current_observation: str = []
        self.plan = None
        self.plan_steps = []
        self.completed_steps = []

    def add_plan(self, plan: str):
        self.plan = plan
        self.save()
    
    def add_learing(self, learn):
        self.learning.append(learn)
        self.save()

    def mark_completed(self):
        if self.plan_steps:
            self.completed_steps.append(self.plan_steps.pop(0))

    def _get_history(self) -> List[Tuple[Type, Role, str]]:
        history_out = []

        ##### GOAL #####
        history_out.append((Type.GOAL, Role.SYSTEM, self._goal))
        ################

        COUNT = 2
        n = COUNT

        # find newest observation in the whole history
        current_observation = None
        recent_observations = []

        if self._history:
            elem = self._history.copy().pop()
            t,r,d = elem
            if t == Type.PERCEPTION:
                current_observation = elem


        for x in reversed(self._history):
            t, r, d = x
            recent_observations.append(x)
            if t == Type.PERCEPTION:
                n = n - 1
            if n <= 0:
                break
        
        recent_observations.reverse()

        mini = []
        for x in self._history:
            if x not in recent_observations and not current_observation:  # exclude the recent observations
                t, r, d = x
                if t != Type.PERCEPTION:
                    mini.append(x)

        ##### MINI HISTORY #####
        if mini:
            history_out.extend(mini)
        ########################

        if current_observation:
            recent_observations.remove(current_observation)

        ##### FULL HISTORY #####
        if recent_observations:
            history_out.extend(recent_observations)
        ########################

        ##### CURRENT OBSERVATION #####
        if current_observation:
            type, role, data = current_observation
            history_out.append((Type.CURRENT_OBSERVATION ,role,"CURRENT OBSERVATION: " + data))
        ###############################

        #learning

        ##### PLAN #####
        if self.completed_steps:
            history_out.append((Type.PLAN, Role.USER, "COMPLETED SUB-GOALS: "+ str(self.completed_steps)))

        if self.plan:
            history_out.append((Type.PLAN, Role.USER, self.plan))
        ################

        #memories

        return history_out

    def get_history(self) -> List[dict[str, str]]:
        max_allowed = int(config.Backend._n_context - max(500, config.Backend._n_context * 0.5))

        print(f"mem: {self.get_token_count()}/{max_allowed} ({config.Backend._n_context})")
        self.assure_max_token_count(max_allowed)
        history_out = self._get_history()

        return super()._build_history(history_out)
    
    def _store(self, path: str) -> None:
        history_out = self._get_history()
        self._save(path, history_out)

    def get_token_count(self) -> int:
        history_out = self._get_history()
        return Memory._approximate_token_count(str(history_out))
    
    def copy(self) -> "SuperMemory":
        new_copy = SuperMemory(self._goal, self.path)
        new_copy._history = copy.deepcopy(self._history)
        new_copy.plan = copy.deepcopy(self.plan)
        new_copy._learnings = copy.deepcopy(self._learnings)
        new_copy._memories = copy.deepcopy(self._memories)
        new_copy._current_observation = copy.deepcopy(self._current_observation)
        return new_copy
