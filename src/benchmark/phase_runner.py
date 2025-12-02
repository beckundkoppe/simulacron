from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from benchmark.dispatcher import Dispatcher
from benchmark.phase_settings import resolve_runner_config
from benchmark.run_registry import (
    build_run,
    filter_by_models,
    normalize_model_teams,
    parse_filename,
)


def _is_git_repo(repo: Path) -> bool:
    return (repo / ".git").exists()


def _current_branch(repo: Path) -> str | None:
    """
    Return the current branch name or None if detached/unknown.
    """
    if not _is_git_repo(repo):
        return None

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else None


def git(repo: Path, *args: str):
    if not _is_git_repo(repo):
        return
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def git_commit_if_needed(repo: Path, paths: List[Path], message: str):
    if not _is_git_repo(repo):
        return
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
    branch = _current_branch(repo)
    if not branch:
        print(f"Skipping git pull; no branch checked out in {repo}")
        return
    git(repo, "pull", "--rebase", "--autostash")


def git_push(repo: Path):
    branch = _current_branch(repo)
    if not branch:
        print(f"Skipping git push; no branch checked out in {repo}")
        return
    git(repo, "push")


def load_todo(phase_file: Path) -> List[str]:
    if not phase_file.exists():
        raise FileNotFoundError(f"Phase file not found: {phase_file}")
    return [line.strip() for line in phase_file.read_text().splitlines() if line.strip()]


def write_todo(phase_file: Path, entries: List[str]):
    phase_file.parent.mkdir(parents=True, exist_ok=True)
    phase_file.write_text("\n".join(entries) + ("\n" if entries else ""))


def claim_entry(claims_folder: Path, entry: str, hostname: str) -> Path:
    """
    Create a claim file for the given entry. File name no longer contains the
    hostname; instead it is written inside the file for bookkeeping.
    """
    claim_path = claims_folder / f"{entry}.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    if claim_path.exists():
        existing_owner = claim_path.read_text().strip()
        # If we already own it, just reuse; otherwise raise to signal contention.
        if existing_owner != hostname:
            raise FileExistsError(f"Claim already exists for {entry}")
        return claim_path

    claim_path.write_text(hostname)
    return claim_path


def cleanup_stale_claims(claims_folder: Path, results_folder: Path, todo_entries: List[str]) -> None:
    claims_folder.mkdir(parents=True, exist_ok=True)
    todo_set = set(todo_entries)
    for claim_file in claims_folder.glob("*.claim"):
        entry = claim_file.stem
        result_file = results_folder / entry
        # If the result exists or the entry is no longer in TODO, drop the claim.
        if result_file.exists() or entry not in todo_set:
            claim_file.unlink(missing_ok=True)
            

def find_local_claim(claims_folder: Path, hostname: str, todo_entries: List[str]) -> Tuple[str, Path] | None:
    todo_set = set(todo_entries)
    for claim_file in claims_folder.glob("*.claim"):
        entry = claim_file.stem
        if entry not in todo_set:
            continue
        owner = claim_file.read_text().strip()
        if owner == hostname:
            return entry, claim_file
    return None


def main():
    results_root = Path(os.getenv("RESULTS_ROOT", "results")).resolve()
    repo_root = Path(os.getenv("RESULTS_GIT_ROOT", results_root)).resolve()
    if not _is_git_repo(repo_root):
        print(f"Git repo not found at {repo_root}, running without pushes.")
    data_root = results_root
    claims_folder = results_root / "runs/claims"
    results_folder = results_root / "runs"
    requested_phase = sys.argv[1] if len(sys.argv) > 1 else None
    runner_config = resolve_runner_config(phase=requested_phase)
    phase = runner_config.phase
    phase_file = data_root / "phase" / f"{phase}.txt"
    allowed_model_teams = normalize_model_teams(None, runner_config.allowed_model_teams or [])
    hostname = socket.gethostname()

    dispatcher = Dispatcher()

    while True:
        git_pull_rebase(repo_root)
        todo_entries = load_todo(phase_file)
        if not todo_entries:
            print("TODO list is empty. Done.")
            break

        cleanup_stale_claims(claims_folder, results_folder, todo_entries)

        # Resume existing claim for this host if present.
        local_claim = find_local_claim(claims_folder, hostname, todo_entries)
        if local_claim:
            claimed_entry, claim_file = local_claim
            print(f"Resuming claimed entry: {claimed_entry}")
            candidates = [claimed_entry]
        else:
            candidates = filter_by_models(todo_entries, allowed_model_teams)
        if not candidates:
            print("No runnable entries for the allowed model teams. Sleeping.")
            break

        claimed_entry = None
        claim_file = None

        for entry in candidates:
            if local_claim and entry == claimed_entry:
                claimed_entry = entry
                break
            try:
                claim_file = claim_entry(claims_folder, entry, hostname)
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
