#!/usr/bin/env bash
set -euo pipefail

configs=(
  "config/lab_lightgbm_top40.yaml"
  "config/lab_lightgbm_top80.yaml"
  "config/lab_lightgbm_top120.yaml"
)

for config in "${configs[@]}"; do
  echo "============================================================"
  echo "Running ${config}"
  echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
  python main.py --config "${config}"
  echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo "All LightGBM top-feature experiments completed."
