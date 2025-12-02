#!/usr/bin/env bash
set -euo pipefail

# Generate phase TODO files with the project virtual environment active.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow running from repo root or from the script/ folder.
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    ROOT_DIR="${SCRIPT_DIR}"
else
    ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
# Default results location inside the code repo; can override via RESULTS_ROOT.
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results}"
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
python -m benchmark.phase_generate "$@"
