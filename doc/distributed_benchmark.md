# Distributed benchmark workflow

This project includes a lightweight git-based flow for coordinating benchmark runs across multiple machines. One machine prepares a **phase TODO list**, and each device claims work from that list until it is empty.

## Prerequisites
- A git repo that contains `data/phase/` and `data/runs/` (can be the main repo, a worktree, or a separate results repo). All devices must pull/push to the same remote.
- Python environment that can run the project (`PYTHONPATH=src`).
- Optional: per-device model allowlist so each runner picks only the models it can execute.

## 1) Generate a phase TODO file
Define the combinations you want to run in Python so you can use enums for levels/models/configurations. Edit `benchmark/phase_settings.py` and adjust the curated `PHASES` list and optional per-host `RUNNER_CONFIGS` mapping:

```python
from benchmark.phase_settings import PHASES, PhaseDefinition, RUNNER_CONFIGS, RunnerConfig
from benchmark.run_registry import CONFIGURATIONS
from enviroment.levels.level import Levels
from llm.model import Model

PHASES = (
    PhaseDefinition(
        phase="phase1",
        configs=[CONFIGURATIONS["baseline"]],
        levels=[Levels.DETAILED_INSTRUCT.value.CARROT_EASY],
        models=[Model.Local.value.LlamaCpp.value.Deepseek.value.CODER_V2_16B_Q8],
        reruns=1,
    ),
)

RUNNER_CONFIGS = {
    "machine-a": RunnerConfig(
        phase="phase1",
        allowed_models=[Model.Local.value.LlamaCpp.value.Deepseek.value.CODER_V2_16B_Q8],
    ),
    # "machine-b": RunnerConfig(
    #     phase="phase1",
    #     allowed_models=[Model.Local.value.LlamaCpp.value.Deepseek.value.CODER_V2_16B_Q4],
    # ),
}
```

Then generate all TODO lists defined in `PHASES`:

```bash
PYTHONPATH=src python -m benchmark.phase_generate
```

The command writes `data/phase/<phase>.txt`, where each line is a result filename (e.g., `carrot-easy-detailed_Deepseek-Coder-V2-16B-Q8_baseline_0.json`). Commit and push this file so all machines see the same TODO list.

## 2) Run benchmarks on a device
Each machine repeatedly pulls the TODO file, claims a runnable entry, executes it, and pushes the updated state. Per-device constraints come from the `RUNNER_CONFIGS` mapping in `benchmark/phase_settings.py`; `benchmark.phase_runner` picks the entry matching the machine hostname (or falls back to the default phase/models from `PHASES`).

```bash
# Runs until no eligible TODO entries remain, using the Python configuration only
PYTHONPATH=src python -m benchmark.phase_runner
```

What happens under the hood:
1. `benchmark.phase_runner` pulls with `git pull --rebase --autostash`.
2. It filters TODO entries by `ALLOWED_MODELS` and tries to create a claim file under `data/runs/claims/`. Successful claims are committed and pushed immediately.
3. The claimed entry is converted back into a `Run` and executed via `Dispatcher.benchmark_single_rerun`.
4. Upon success, the TODO line is removed, the result JSON is written to `data/runs/`, the claim is removed, and the changes are committed and pushed.

If a claim fails because another device won the race, the runner tries the next entry. When the TODO file is empty (or no entries match the allowed models), the process exits.

## 3) Operational tips
- Pull before each session: `git pull --rebase`.
- Keep commits small and frequent so claims/results propagate quickly.
- Claims live in `data/runs/claims/` and are cleaned up automatically after successful runs; stale claims can be deleted manually if a device crashes.
- The TODO file and results are the single source of truth—if a result JSON exists, the dispatcher automatically skips duplicate runs.
