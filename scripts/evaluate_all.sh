#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/evaluate_all.sh GRASP_DIRECTORY" >&2
    exit 2
fi

grasp_dir="$1"
found=0
for robot in barrett allegro shadowhand leaphand inspire; do
    if [ -d "$grasp_dir/$robot" ]; then
        python scripts/evaluate.py --input "$grasp_dir/$robot" --robot "$robot"
        found=1
    fi
done
if [ "$found" -eq 0 ]; then
    echo "No hand directories found in $grasp_dir" >&2
    exit 1
fi
