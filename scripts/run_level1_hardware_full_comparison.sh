#!/usr/bin/env bash
set -euo pipefail

config="config/level1_hardware_full_comparison.yaml"

echo "============================================================"
echo "Level 1 hardware full comparison"
echo "Config: ${config}"
echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
python main.py --config "${config}"
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"
