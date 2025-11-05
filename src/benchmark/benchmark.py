from datetime import datetime
import time
from datetime import datetime
from typing import Dict, TypedDict

from llama_cpp import List
from benchmark.result import Result
from llm.model import Model

from benchmark.configuration import Configuration, get_sets

AVG_SINGLE_RUN_SECONDS = 120.0

REPEAT_COUNT = 3

def full_run() -> Dict[str, Dict[str, Result]]:
    all_results: Dict[str, Dict[str, Result]] = {}
    config_sets = get_sets()

    # ---- Gesamt-Laufzeitschätzung vorbereiten ----
    total_runs = 0
    for set_data in config_sets.values():
        configs = set_data["configs"]
        models = set_data["models"]
        total_runs += len(configs) * len(models) * REPEAT_COUNT

    total_expected_seconds = total_runs * AVG_SINGLE_RUN_SECONDS
    total_expected_minutes = total_expected_seconds / 60
    total_expected_hours = total_expected_minutes / 60

    print(f"\n=== BENCHMARK START ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"  → Total expected runs: {total_runs}")
    print(f"  → Estimated total runtime: "
          f"{total_expected_minutes:.1f} min ({total_expected_hours:.2f} h)\n")

    # ---- Haupt-Schleife über Sets ----
    for set_name, set_data in config_sets.items():
        configs = set_data["configs"]
        models = set_data["models"]

        # --- Estimate expected runtime per set ---
        num_runs = len(configs) * len(models) * REPEAT_COUNT
        expected_seconds = num_runs * AVG_SINGLE_RUN_SECONDS
        expected_minutes = expected_seconds / 60
        expected_hours = expected_minutes / 60

        print(f"\n--- Set: {set_name} ---")
        print(f"  Configs: {len(configs)} | Models: {len(models)} | Repeat: {REPEAT_COUNT}")
        print(f"  → Expected runs: {num_runs}")
        print(f"  → Estimated runtime: "
              f"{expected_minutes:.1f} min ({expected_hours:.2f} h)\n")

        # --- Execute benchmark set ---
        for config in configs:
            print(f"  → Running config: {config.name}")
            t0 = time.time()

            results_for_config: Dict[str, Result] = {}
            for model in models:
                results_for_config[model.tag] = single_run(config, model)

            t1 = time.time()
            duration = t1 - t0

            avg_sr = sum(r.success_rate for r in results_for_config.values()) / len(results_for_config)
            print(f"     avg success rate: {avg_sr:.2%} | runtime: {duration:.1f}s")

            all_results[config.name] = results_for_config

    print(f"\n=== BENCHMARK COMPLETE ({len(all_results)} configurations total) ===")
    return all_results

def single_run(config: Configuration, model: "Model") -> Result:
    run_results: List[Result] = []

    for _ in range(REPEAT_COUNT):
        result = None #TODO
        run_results.append(result)

    return Result.average(run_results)