#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python -m venv venv
fi

if [ -z "$VIRTUAL_ENV" ]; then
  source venv/bin/activate
  # Check if requirements are satisfied, install quietly if not
  if ! pip check > /dev/null 2>&1; then
    pip install --quiet -r requirements.txt
  fi
fi

python script/setup.py
