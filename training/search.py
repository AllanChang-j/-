from __future__ import annotations

import itertools
import random
from typing import Any, Callable


def grid_search(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    values = [space[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def random_search(space: dict[str, list[Any]], max_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(space)
    trials = []
    for _ in range(max_trials):
        trials.append({key: rng.choice(space[key]) for key in keys})
    return trials


def optuna_search(
    objective: Callable[[dict[str, Any]], float],
    space: dict[str, list[Any]],
    max_trials: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is not installed. Install optuna or use grid/random search.") from exc

    def optuna_objective(trial: Any) -> float:
        params = {key: trial.suggest_categorical(key, values) for key, values in space.items()}
        return objective(params)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(optuna_objective, n_trials=max_trials)
    return study.best_params, float(study.best_value)


def candidate_params(
    method: str,
    base_params: dict[str, Any],
    space: dict[str, list[Any]] | None,
    max_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    if method == "none" or not space:
        return [base_params]
    if method == "grid":
        return [{**base_params, **params} for params in grid_search(space)]
    if method == "random":
        return [{**base_params, **params} for params in random_search(space, max_trials=max_trials, seed=seed)]
    if method == "optuna":
        return [base_params]
    raise ValueError(f"Unsupported hyperparameter search method: {method}")
