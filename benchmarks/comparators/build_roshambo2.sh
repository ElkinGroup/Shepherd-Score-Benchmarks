#!/usr/bin/env bash
set -euo pipefail
: "${ROSHAMBO2_SOURCE:?Set ROSHAMBO2_SOURCE to an empty directory for the source checkout}"
if [[ -e "$ROSHAMBO2_SOURCE" ]]; then
    echo "Choose a new ROSHAMBO2_SOURCE directory" >&2
    exit 1
fi
git clone https://github.com/molecularinformatics/roshambo2.git "$ROSHAMBO2_SOURCE"
git -C "$ROSHAMBO2_SOURCE" checkout --detach 7ff8237c4f21372e8fbfb4e842b1b5eee87a7aeb
python -m pip install --no-build-isolation "$ROSHAMBO2_SOURCE"
python -c 'from roshambo2 import Roshambo2'
