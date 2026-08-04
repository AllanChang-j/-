#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/quick_ta_regression_deep.yaml}"

if [[ ! -x ".venv/bin/python" ]]; then
  CONFIG_PATH="${CONFIG_PATH}" bash scripts/lab_one_click_level1.sh
else
  .venv/bin/python main.py --config "${CONFIG_PATH}"
fi
