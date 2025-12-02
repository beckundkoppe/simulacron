import contextlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List

import config
import current
import game
from benchmark.benchresult import PerformanceResult
from benchmark.model_team import ModelTeam
from benchmark.run import Run
from enviroment.levels.level import Level
from enviroment.world import World

class Dispatcher:
    def __init__(self):
        self.queued_runs: list[Run] = []
        self.average_time = 30.0
        results_root = Path(os.getenv("RESULTS_ROOT", "results"))
        self.folder = str(results_root / "runs/")
        self.folder_phase = str(results_root / "phase/")

    def queue_run(self, run: Run):
        self.queued_runs.append(run)

    @staticmethod
    def _basename_for_run(run: Run, rerun_index: int) -> str:
        team_token = run.model_team.token()
        return f"{run.level.value.getName()}_{team_token}_{run.configuration.name}_{rerun_index}"

    @staticmethod
    def _raw_log_path(base_folder: str, basename: str) -> Path:
        raw_folder = Path(base_folder) / "raw"
        raw_folder.mkdir(parents=True, exist_ok=True)
        return raw_folder / f"{basename}_raw.txt"

    @contextlib.contextmanager
    def _redirect_output_to_raw(self, run: Run, rerun_index: int):
        basename = self._basename_for_run(run, rerun_index)
        raw_path = self._raw_log_path(self.folder, basename)
        with raw_path.open("w") as raw_file:
            tee = _TeeStream(sys.stdout, raw_file)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                yield raw_path

    def _log_run_failure(self, run: Run, rerun_index: int, error: Exception) -> None:
        """Persist failure details into the run's result file."""
        try:
            basename = self._basename_for_run(run, rerun_index)
            os.makedirs(self.folder, exist_ok=True)
            path = Path(self.folder) / f"{basename}.json"
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry": basename,
                "level": run.level.value.getName(),
                "config": run.configuration.name,
                "model_team": run.model_team.token(),
                "rerun_index": rerun_index,
                "status": "error",
                "error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            }
            path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        except Exception:
            # Never let logging failure break execution flow.
            pass


    def run_single(self, run: Run):
        results = []

        config.ACTIVE_CONFIG = run.configuration
        for i in range(run.reruns):
            with self._redirect_output_to_raw(run, i):
                print(f"Rerun: {i+1}")

                try:
                    result = self._start_with_result(run)
                except Exception as e:
                    self._log_run_failure(run, i, e)
                    raise
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
        for i in range(run.reruns):
            # Reuse the rerun helper so claimed single runs and full runs share logic
            self.benchmark_single_rerun(run, i)



    def benchmark_matrix(self,
                     configs: List[config.Configuration],
                     levels: List[Level],
                    model_teams: List[ModelTeam],
                    reruns: int,
                    phase: str | None = None # without .txt
                    ):
        if phase:
            self.matrix_generate(configs, levels, model_teams, reruns, phase)

        for team, conf, lvl in product(model_teams, configs, levels):
            run = Run(
                configuration=conf,
                model_team=team,
                level=lvl,
                reruns=reruns,
                optimal_steps_multiplier=2.0,   # falls nötig
            )
            self.benchmark_single(run)


    def matrix_generate(self,
                        configs: List[config.Configuration],
                        levels: List[Level],
                        model_teams: List[ModelTeam],
                        reruns: int,
                        phase: str,
                        ) -> list[str]:
        todo_entries: list[str] = []

        for team, conf, lvl in product(model_teams, configs, levels):
            run = Run(
                configuration=conf,
                model_team=team,
                level=lvl,
                reruns=reruns,
                optimal_steps_multiplier=2.0,
            )
            for i in range(run.reruns):
                todo_entries.append(self._basename_for_run(run, i) + ".json")

        os.makedirs(self.folder_phase, exist_ok=True)
        path = os.path.join(self.folder_phase, phase + ".txt")
        self._write_file(path, "\n".join(todo_entries) + ("\n" if todo_entries else ""))
        return todo_entries


    def benchmark_single_rerun(self, run: Run, rerun_index: int):
        os.makedirs(self.folder, exist_ok=True)

        config.ACTIVE_CONFIG = run.configuration

        basename = self._basename_for_run(run, rerun_index)
        filename = basename + ".json"
        path = os.path.join(self.folder, filename)

        if os.path.exists(path):
            return

        with self._redirect_output_to_raw(run, rerun_index):
            print(f"Starting: {filename}")
            try:
                result = self._start_with_result(run)
                self._write_file(path, result.toJSON())
                print(result.toString())
            except Exception as e:
                self._log_run_failure(run, rerun_index, e)
                raise


    # def _debug_result(self, run,i):
    #     return PerformanceResult(run, 1, i, 8, 4, 1, 1, 7, 100)


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


class _TeeStream:
    """
    Minimal stream duplicator so output still appears in the console while also
    being captured into the raw log file.
    """

    def __init__(self, *targets):
        self.targets = targets

    def write(self, data):
        for target in self.targets:
            target.write(data)

    def flush(self):
        for target in self.targets:
            target.flush()
