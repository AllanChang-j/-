#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-.venv}"
CONFIG_PATH="${CONFIG_PATH:-config/level1_hardware_full_comparison.yaml}"
DATA_PATH="${DATA_PATH:-data/raw/taiwan_daily_ohlcv_20240101_20260630.csv}"
DATA_START="${DATA_START:-2024-01-01}"
DATA_END="${DATA_END:-2026-06-30}"
REQUEST_DELAY="${REQUEST_DELAY:-0.25}"
COLLECT_IF_MISSING="${COLLECT_IF_MISSING:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-auto}"
CUDA_TORCH_INDEX_URL="${CUDA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${LOG_DIR}" data/raw experiments
export CONFIG_PATH DATA_PATH
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LOG_DIR}/level1_one_click_${RUN_ID}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

find_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo ""
}

main() {
  log "Project root: ${ROOT_DIR}"
  log "Log path: ${LOG_PATH}"
  log "Config: ${CONFIG_PATH}"
  log "Data: ${DATA_PATH}"

  local python_cmd
  python_cmd="$(find_python)"
  if [[ -z "${python_cmd}" ]]; then
    echo "Python was not found. Install Python 3.11 or 3.12 first, then rerun this script." >&2
    exit 1
  fi

  log "Using Python: ${python_cmd}"
  "${python_cmd}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  log "Upgrading pip tooling"
  python -m pip install --upgrade pip setuptools wheel

  if [[ "${INSTALL_CUDA_TORCH}" == "1" ]] || { [[ "${INSTALL_CUDA_TORCH}" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; }; then
    log "NVIDIA GPU detected or requested; installing CUDA PyTorch from ${CUDA_TORCH_INDEX_URL}"
    python -m pip install --upgrade torch torchvision torchaudio --index-url "${CUDA_TORCH_INDEX_URL}"
  else
    log "Skipping explicit CUDA PyTorch install"
  fi

  log "Installing research dependencies"
  python -m pip install -r requirements-research.txt

  log "Environment report"
  python scripts/verify_lab_environment.py

  if [[ ! -f "${DATA_PATH}" ]]; then
    if [[ "${COLLECT_IF_MISSING}" != "1" ]]; then
      echo "Missing ${DATA_PATH}. Copy the CSV into data/raw or set COLLECT_IF_MISSING=1." >&2
      exit 1
    fi
    log "Data file missing; collecting official daily history ${DATA_START} to ${DATA_END}"
    python data/collect_stage1_history.py \
      --start "${DATA_START}" \
      --end "${DATA_END}" \
      --output "${DATA_PATH}" \
      --request-delay "${REQUEST_DELAY}" \
      --strict-network
  else
    log "Data file exists; skipping collection"
  fi

  log "Validating daily data file"
  python - <<'PY'
import os
from data.loaders import load_daily_csv
path = os.environ["DATA_PATH"]
df = load_daily_csv(path)
print("rows", len(df))
print("date_min", df["date"].min().date())
print("date_max", df["date"].max().date())
print("symbols", df["symbol"].nunique())
print("markets", df["market"].value_counts(dropna=False).to_dict())
PY

  if [[ "${RUN_TESTS}" == "1" ]]; then
    log "Running unit safety tests"
    python -m pytest tests
  fi

  log "Starting Level 1 training and final test"
  python main.py --config "${CONFIG_PATH}"

  log "Level 1 run completed"
  log "Key outputs:"
  python - <<'PY'
import os
from pathlib import Path
from utils.config import load_config
cfg = load_config(os.environ["CONFIG_PATH"])
out = Path(cfg["experiment"]["output_dir"])
for rel in [
    "reports/final_comparison_report.xlsx",
    "reports/test_comparison.csv",
    "reports/backtest_comparison.csv",
    "reports/cross_validation_summary.csv",
    "reports/best_model_summary.json",
]:
    path = out / rel
    print(path, "exists" if path.exists() else "missing")
PY
}

main "$@" 2>&1 | tee "${LOG_PATH}"
