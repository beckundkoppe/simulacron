#!/usr/bin/env bash
set -euo pipefail

# Wrapper to generate phase TODO lists and run the phase runner with the project venv active.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow running from repo root or from the script/ folder.
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    ROOT_DIR="${SCRIPT_DIR}"
else
    ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# Optional override to keep results in a separate repo/worktree.
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results}"
RESULTS_GIT_ROOT="${RESULTS_GIT_ROOT:-$RESULTS_ROOT}"
PHASE_NAME="${1:-}"

cd "$ROOT_DIR"

# Ensure venv exists.
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv if not already active.
if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "${ROOT_DIR}/venv" ]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/venv/bin/activate"
fi

# Install/upgrade dependencies quietly.
pip install -q --upgrade pip
pip install -q -r requirements.txt

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export RESULTS_ROOT
export RESULTS_GIT_ROOT

# Ensure TODO files are up to date from git.
if [ -d "${RESULTS_ROOT}/.git" ]; then
    git -C "$RESULTS_ROOT" pull --rebase --autostash
fi

mkdir -p "$RESULTS_ROOT"

# Decide which phases to run: explicit argument or all *.txt files in results/phase.
if [[ -n "$PHASE_NAME" ]]; then
    PHASES=("$PHASE_NAME")
else
    PHASES=()
    if compgen -G "${RESULTS_ROOT}/phase/*.txt" > /dev/null; then
        while IFS= read -r -d '' file; do
            phase_name="$(basename "${file%.txt}")"
            PHASES+=("$phase_name")
        done < <(find "${RESULTS_ROOT}/phase" -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)
    fi
fi

if [[ ${#PHASES[@]} -eq 0 ]]; then
    echo "No phases found to run (results/phase/*.txt is empty)."
    exit 0
fi

echo "Starting phase runner for phases: ${PHASES[*]}"
# Run from the results repo so results/phase and results/runs live there.
for phase in "${PHASES[@]}"; do
    echo "Running phase '${phase}'..."
    python -m benchmark.phase_runner "$phase"
done
