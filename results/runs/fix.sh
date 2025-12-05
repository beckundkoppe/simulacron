#!/usr/bin/env bash
# Renames files by replacing "plan-agent" with "a-agent" in the filename.

for f in *plan-agent*.json; do
    # Skip if no match
    [ -e "$f" ] || continue

    # Build new name
    new="${f/plan-agent/a-agent}"

    # Rename
    mv "$f" "$new"
done
