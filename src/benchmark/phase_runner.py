from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

from benchmark.dispatcher import Dispatcher
from benchmark.phase_settings import resolve_runner_config
from benchmark.run_registry import (
    build_run,
    filter_by_models,
    normalize_model_teams,
    parse_filename,
)


CLAIMS_FOLDER = Path("data/runs/claims")


def git(repo: Path, *args: str):
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def git_commit_if_needed(repo: Path, paths: List[Path], message: str):
    staged_targets = []
    for path in paths:
        if path is None:
            continue
        target = path if path.exists() else path.parent
        staged_targets.append(str(target))

    if not staged_targets:
        return

    git(repo, "add", "-A", *staged_targets)
    # Only commit when there is something staged
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        return
    git(repo, "commit", "-m", message)


def git_pull_rebase(repo: Path):
    git(repo, "pull", "--rebase", "--autostash")


def git_push(repo: Path):
    git(repo, "push")


def load_todo(phase_file: Path) -> List[str]:
    if not phase_file.exists():
        raise FileNotFoundError(f"Phase file not found: {phase_file}")
    return [line.strip() for line in phase_file.read_text().splitlines() if line.strip()]


def write_todo(phase_file: Path, entries: List[str]):
    phase_file.parent.mkdir(parents=True, exist_ok=True)
    phase_file.write_text("\n".join(entries) + ("\n" if entries else ""))


def claim_entry(repo_root: Path, entry: str) -> Path:
    claim_path = repo_root / CLAIMS_FOLDER / f"{entry}.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    if claim_path.exists():
        raise FileExistsError(f"Claim already exists for {entry}")
    claim_path.write_text("")
    return claim_path


def main():
    repo_root = Path.cwd()
    requested_phase = sys.argv[1] if len(sys.argv) > 1 else None
    runner_config = resolve_runner_config(phase=requested_phase)
    phase = runner_config.phase
    phase_file = repo_root / "data/phase" / f"{phase}.txt"
    allowed_model_teams = normalize_model_teams(None, runner_config.allowed_model_teams or [])

    dispatcher = Dispatcher()

    while True:
        git_pull_rebase(repo_root)
        todo_entries = load_todo(phase_file)
        if not todo_entries:
            print("TODO list is empty. Done.")
            break

        candidates = filter_by_models(todo_entries, allowed_model_teams)
        if not candidates:
            print("No runnable entries for the allowed model teams. Sleeping.")
            break

        claimed_entry = None
        claim_file = None

        for entry in candidates:
            try:
                claim_file = claim_entry(repo_root, entry)
            except FileExistsError:
                continue

            try:
                git_commit_if_needed(repo_root, [claim_file], f"claim {entry}")
                git_pull_rebase(repo_root)
                git_push(repo_root)
                claimed_entry = entry
                break
            except subprocess.CalledProcessError:
                # Someone else won the race; try next entry
                claim_file.unlink(missing_ok=True)
                continue

        if not claimed_entry:
            print("No free entries after attempting claims.")
            break

        level_name, model_team, config_name, rerun_index = parse_filename(claimed_entry)
        run = build_run(level_name, model_team, config_name, rerun_index)
        dispatcher.benchmark_single_rerun(run, rerun_index)

        todo_entries = [item for item in todo_entries if item != claimed_entry]
        write_todo(phase_file, todo_entries)

        result_path = repo_root / dispatcher.folder / claimed_entry
        claim_file.unlink(missing_ok=True)

        git_commit_if_needed(repo_root, [phase_file, result_path, claim_file], f"finish {claimed_entry}")
        git_pull_rebase(repo_root)
        git_push(repo_root)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
