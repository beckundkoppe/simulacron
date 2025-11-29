#!/usr/bin/env bash
set -euo pipefail

# Wrapper to generate phase TODO lists and run the phase runner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE_NAME="${1:-}"

cd "$ROOT_DIR"

# Make sure our modules are importable.
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Generating phase TODO files..."
python -m benchmark.phase_generate

echo "Starting phase runner${PHASE_NAME:+ for phase '${PHASE_NAME}'}..."
if [[ -n "$PHASE_NAME" ]]; then
    python -m benchmark.phase_runner "$PHASE_NAME"
else
    python -m benchmark.phase_runner
fi
