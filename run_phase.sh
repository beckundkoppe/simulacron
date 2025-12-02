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

echo "Starting phase runner${PHASE_NAME:+ for phase '${PHASE_NAME}'}..."
# Run from the results repo so results/phase and results/runs live there.
mkdir -p "$RESULTS_ROOT"
if [[ -n "$PHASE_NAME" ]]; then
    python -m benchmark.phase_runner "$PHASE_NAME"
else
    python -m benchmark.phase_runner
fi
