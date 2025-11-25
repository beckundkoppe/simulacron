from dataclasses import dataclass
import time
from benchmark.benchresult import PerformanceResult
from benchmark.run import Run
import current
from enviroment.world import World
import game
import config

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
            result = PerformanceResult(run)
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
            print(PerformanceResult.average(results).toString())
        
        return PerformanceResult.average(results)

    def run_all(self):
        results = []

        for run in self.queued_runs:
            results.append(self.run_single(run))

        return results


    def expected_time(self) -> float:
        return len(self.queued_runs) * self.average_time
