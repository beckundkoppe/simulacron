#!/usr/bin/env bash
# Renames files by replacing "plan-agent" with "a-agent" in the filename.

for f in *a-agent*.json; do
    # Skip if no match
    [ -e "$f" ] || continue

    # Build new name
    new="${f/a-agent/plan-agent}"

    # Rename
    mv "$f" "$new"
done
