from dataclasses import dataclass
import time
from benchmark.benchresult import RunResult
from enviroment import current
from enviroment.world import World
import game
import config
from config import Configuration
from enviroment.levels.level import Level
from llm import cache
from llm.model import Model

@dataclass(frozen=True)
class Run:
    configuration: Configuration
    model: Model
    level: Level
    reruns: int
    optimal_steps_multiplier: float

class Dispatcher:
    def __init__(self, cache):
        self.cache = cache
        self.queued_runs: list[Run] = []
        self.average_time = 30.0

    def queue_run(self, run: Run):
        self.queued_runs.append(run)

    def run_single(self, run: Run):
        results = []

        config.CONFIG = run.configuration
        for i in range(run.reruns):
            print(f"Rerun: {i+1}")
            World.clear()
            result = RunResult(run.model.value.tag, run.configuration.name, run.level.name, run.level.optimal_steps)
            current.RESULT = result

            self.cache.get(run.model) # load model before starting timer

            start_time = time.time()
            game.run_level(self.cache, run.model, run.level, run.optimal_steps_multiplier)
            end_time = time.time()
            result.time_s = end_time - start_time

            print(result.softerror_count)
            print(result.harderror_count)
            results.append(result)
            current.RESULT = None
        
        return RunResult.average(results)

    def run_all(self):
        results = []
        
        for run in self.queued_runs:
            results.append(self.run_single(run))


    def expected_time(self) -> float:
        return self.queue_run.count * self.average_time