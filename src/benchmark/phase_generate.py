from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark.dispatcher import Dispatcher
from benchmark.phase_settings import PHASES
from benchmark.run_registry import (
    normalize_configurations,
    normalize_levels,
    normalize_model_teams,
)


def main():
    repo_root = Path.cwd()
    dispatcher = Dispatcher()

    def _generate(definition) -> tuple[str, Iterable[str]]:
        configs = normalize_configurations(None, definition.configs)
        levels = normalize_levels(None, definition.levels)
        model_teams = normalize_model_teams(None, definition.model_teams)
        todos = dispatcher.matrix_generate(configs, levels, model_teams, definition.reruns, definition.phase)
        return definition.phase, todos

    for phase, todos in (_generate(definition) for definition in PHASES):
        phase_path = repo_root / dispatcher.folder_phase / f"{phase}.txt"
        print(f"Wrote {len(todos)} TODO entries to {phase_path}")


if __name__ == "__main__":
    main()
