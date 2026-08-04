from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"missing ({type(exc).__name__}: {exc})"
    return str(getattr(module, "__version__", "installed"))


def command_output(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return "not found"
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def main() -> None:
    print("=== Lab Environment ===")
    print("python", sys.version.replace("\n", " "))
    print("platform", platform.platform())
    print("executable", sys.executable)
    print("nvidia-smi", command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]))
    print("numpy", module_version("numpy"))
    print("pandas", module_version("pandas"))
    print("sklearn", module_version("sklearn"))
    print("lightgbm", module_version("lightgbm"))
    print("torch", module_version("torch"))
    try:
        import torch

        print("torch.cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("torch.cuda_device", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("torch.cuda_check_error", repr(exc))

    data_path = Path("data/raw/taiwan_daily_ohlcv_20240101_20260630.csv")
    print("data_path", data_path)
    print("data_exists", data_path.exists())
    if data_path.exists():
        print("data_size_mb", round(data_path.stat().st_size / 1024 / 1024, 2))


if __name__ == "__main__":
    main()
