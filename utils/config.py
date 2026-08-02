from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        output_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    current: Any = config
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

