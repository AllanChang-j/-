from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.base import PredictionResult
from preprocessing.windows import WindowDataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task: str):
        self.X = torch.tensor(X, dtype=torch.float32)
        if task == "regression":
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
        self.task = task

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


@dataclass
class TorchTrainingConfig:
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    scheduler: str = "cosine"
    device: str = "cpu"
    num_workers: int = 0


def _loss_for_task(task: str) -> nn.Module:
    if task == "regression":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def _score_for_early_stopping(task: str, validation_loss: float) -> float:
    return -validation_loss


def _predict_from_logits(logits: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray | None]:
    if task == "regression":
        return logits.reshape(-1), None
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return probabilities.argmax(axis=1), probabilities


def _state_dict_size_bytes(model: nn.Module) -> int:
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        return Path(tmp.name).stat().st_size


def train_torch_model(
    model_name: str,
    model: nn.Module,
    train_data: WindowDataset,
    validation_data: WindowDataset,
    task: str,
    config: TorchTrainingConfig,
    output_dir: str | Path,
    params: dict[str, Any],
) -> PredictionResult:
    device = torch.device(config.device)
    model = model.to(device)
    criterion = _loss_for_task(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.max_epochs))
        if config.scheduler == "cosine"
        else None
    )

    train_loader = DataLoader(
        SequenceDataset(train_data.X, train_data.y, task),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        SequenceDataset(validation_data.X, validation_data.y, task),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    best_score = -np.inf
    best_state = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard" / model_name))
    except Exception:
        writer = None

    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            if config.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                validation_losses.append(float(criterion(logits, batch_y).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"{model_name} epoch {epoch + 1}/{config.max_epochs}: "
            f"train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}, "
            f"elapsed_sec={time.perf_counter() - epoch_start:.1f}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", validation_loss, epoch)
        score = _score_for_early_stopping(task, validation_loss)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_time = time.perf_counter() - start_time
    if writer is not None:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    inference_start = time.perf_counter()
    logits_parts = []
    model.eval()
    with torch.no_grad():
        for batch_X, _batch_y in validation_loader:
            batch_X = batch_X.to(device)
            logits_parts.append(model(batch_X).detach().cpu().numpy())
    inference_time = time.perf_counter() - inference_start

    logits = np.concatenate(logits_parts, axis=0)
    predictions, probabilities = _predict_from_logits(logits, task)
    output_path = Path(output_dir) / "models" / f"{model_name}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "params": params, "task": task, "history": history}, output_path)

    return PredictionResult(
        model_name=model_name,
        predictions=predictions,
        probabilities=probabilities,
        targets=validation_data.y,
        meta=validation_data.meta.copy(),
        training_time_sec=training_time,
        inference_time_sec=inference_time,
        model_size_bytes=_state_dict_size_bytes(model),
        history=history,
        params=params,
    )


def predict_torch_model(
    model_name: str,
    model: nn.Module,
    dataset: WindowDataset,
    task: str,
    config: TorchTrainingConfig,
    params: dict[str, Any],
) -> PredictionResult:
    device = torch.device(config.device)
    model = model.to(device)
    loader = DataLoader(
        SequenceDataset(dataset.X, dataset.y, task),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    start = time.perf_counter()
    logits_parts = []
    model.eval()
    with torch.no_grad():
        for batch_X, _batch_y in loader:
            batch_X = batch_X.to(device)
            logits_parts.append(model(batch_X).detach().cpu().numpy())
    inference_time = time.perf_counter() - start
    logits = np.concatenate(logits_parts, axis=0)
    predictions, probabilities = _predict_from_logits(logits, task)
    return PredictionResult(
        model_name=model_name,
        predictions=predictions,
        probabilities=probabilities,
        targets=dataset.y,
        meta=dataset.meta.copy(),
        inference_time_sec=inference_time,
        model_size_bytes=_state_dict_size_bytes(model),
        params=params,
    )
