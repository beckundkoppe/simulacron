#!/usr/bin/env bash
set -euo pipefail

# Create venv folder if it does not exist yet
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv if we are not already inside it
if [ -z "${VIRTUAL_ENV:-}" ] || [ "$VIRTUAL_ENV" != "$(pwd)/venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Upgrade pip and install required dependencies (quiet mode)
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run the application
python src/main.py
