from dataclasses import dataclass
import time
from benchmark.benchresult import RunResult
import current
from enviroment.world import World
import game
import config
from config import Configuration
from enviroment.levels.level import Level
from llm.model import Model

@dataclass(frozen=True)
class Run:
    configuration: Configuration
    main_model: Model
    level: Level
    reruns: int
    optimal_steps_multiplier: float
    imaginator: Model = None
    extra_model: Model = None

class Dispatcher:
    def __init__(self):
        self.queued_runs: list[Run] = []
        self.average_time = 30.0

    def queue_run(self, run: Run):
        self.queued_runs.append(run)

    def run_single(self, run: Run):
        results = []

        config.ACTIVE_CONFIG = run.configuration
        for i in range(run.reruns):
            print(f"Rerun: {i+1}")
            World.clear()
            result = RunResult(run.main_model.value.tag, run.configuration.name, run.level.name, run.level.optimal_steps)
            current.RESULT = result

            #self.cache.get(run.main_model) # load model before starting timer

            #if(run.imaginator is not None):
                #self.cache.get(run.imaginator)

            start_time = time.time()
            game.run_level(run.level, run.optimal_steps_multiplier, run.main_model, run.imaginator, run.extra_model)
            end_time = time.time()
            result.time_s = end_time - start_time

            print(result.softerror_count)
            print(result.harderror_count)
            results.append(result)
            current.RESULT = None
            print(RunResult.average(results).toString())
        
        return RunResult.average(results)

    def run_all(self):
        results = []
        
        for run in self.queued_runs:
            results.append(self.run_single(run))


    def expected_time(self) -> float:
        return self.queue_run.count * self.average_time