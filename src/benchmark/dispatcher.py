from dataclasses import dataclass
from itertools import product
import time
from typing import List
from benchmark.benchresult import PerformanceResult
from benchmark.run import Run
import current
from enviroment.levels.level import Level
from enviroment.world import World
import game
import config
import os

from llm.model import Model

class Dispatcher:
    def __init__(self):
        self.queued_runs: list[Run] = []
        self.average_time = 30.0
        self.folder = "data/runs/"
        self.folder_phase = "data/phase/"
        self.file = None

    def queue_run(self, run: Run):
        self.queued_runs.append(run)


    def append_raw(self, str: str) -> None:
        folder_raw = os.path.join(self.folder, "raw/")
        os.makedirs(folder_raw, exist_ok=True)
        filename_raw = os.path.join(folder_raw, self.file+"_raw.txt")
        with open(filename_raw, "a") as f: # -> nur am ende dranhängen
            f.write(str)


    def run_single(self, run: Run):
        results = []

        config.ACTIVE_CONFIG = run.configuration
        for i in range(run.reruns):
            print(f"Rerun: {i+1}")

            result = self._start_with_result(run)
            print(result.softerror_count)
            print(result.harderror_count)
            
            results.append(result)
            current.RESULT = None
            print(PerformanceResult.average(results).toString())
        
        return PerformanceResult.average(results)

    def _start_with_result(self, run: Run) -> PerformanceResult:
        config.ACTIVE_CONFIG = run.configuration
        World.clear()
        result = PerformanceResult(run)
        current.RESULT = result
        start_time = time.time()
        game.run_level(run.level, run.optimal_steps_multiplier, run.main_model, run.imaginator, run.extra_model)
        end_time = time.time()
        result.time_s = end_time - start_time
        current.RESULT = None
        World.clear()
        config.ACTIVE_CONFIG = None
        return result


    def benchmark_single(self, run: Run):
        os.makedirs(self.folder, exist_ok=True)
        
        config.ACTIVE_CONFIG = run.configuration
        config.APPEND_RAW = self.append_raw
        for i in range(run.reruns):
            self.file = f"{run.level.value.getName()}_{run.main_model.value.name}_{run.configuration.name}_{i}"
            filename = self.file + ".json"
            path = os.path.join(self.folder, filename)

            if os.path.exists(path):
                continue

            print(f"Starting: {filename}")
            result = self._start_with_result(run)
            #result = self._debug_result(run, i)
            
            self._write_file(path, result.toJSON())

            print(result.toString())



    def benchmark_matrix(self,
                     configs: List[config.Configuration],
                     levels: List[Level],
                     models: List[Model],
                     reruns: int,
                     phase: str | None = None # without .txt
                     ):
        if phase:
            write = ""
            for mdl, conf, lvl in product(models, configs, levels):
                run = Run(
                    configuration=conf,
                    main_model=mdl,
                    level=lvl,
                    reruns=reruns,
                    optimal_steps_multiplier=1.0,   # falls nötig
                    imaginator=None,                # falls du das setzen willst
                    extra_model=None
                )
                for i in range(run.reruns):
                    content = f"{run.level.value.getName()}_{run.main_model.value.name}_{run.configuration.name}_{i}.json"
                    write += content + "\n"

            os.makedirs(self.folder_phase, exist_ok=True)
            path = os.path.join(self.folder_phase, phase+".txt")
            self._write_file(path, write)



        for mdl, conf, lvl in product(models, configs, levels):
            run = Run(
                configuration=conf,
                main_model=mdl,
                level=lvl,
                reruns=reruns,
                optimal_steps_multiplier=1.0,   # falls nötig
                imaginator=None,                # falls du das setzen willst
                extra_model=None
            )
            self.benchmark_single(run)
        
        
        
        for mdl, conf, lvl in product(models, configs, levels):
            run = Run(
                configuration=conf,
                main_model=mdl,
                level=lvl,
                reruns=reruns,
                optimal_steps_multiplier=1.0,   # falls nötig
                imaginator=None,                # falls du das setzen willst
                extra_model=None
            )

            self.benchmark_single(run)

        
    def _debug_result(self, run,i):
        config.APPEND_RAW("Hello World")
        return PerformanceResult(run, 1, i, 8, 4, 1, 1, 7, 100)


    def run_all(self):
        results = []

        for run in self.queued_runs:
            results.append(self.run_single(run))

        return results


    def expected_time(self) -> float:
        return len(self.queued_runs) * self.average_time


    def _write_file(self, path, content):
        with open(path, "w") as f:
            f.write(content)
