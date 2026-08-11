import json
import os
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from log_utils import _log
from preprocessing import build_preprocessor
from utils import model_filename, run_method


def _nn_predict(model: nn.Module, X: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Выполняет предсказание нейросети батчами."""
    model.eval()
    preds = []

    dataset = TensorDataset(X)
    loader = DataLoader(dataset=dataset, batch_size=batch_size)

    with torch.no_grad():
        for (batch_X,) in loader:
            outputs = model(batch_X)
            preds.append(outputs.squeeze().numpy())

    return np.concatenate(preds)


def save_nn_submission(
    model: nn.Module,
    X_submit: pd.DataFrame,
    cfg: DictConfig,
    preproc: Any,
    submission_name: str = "NN_Model",
) -> str:
    """Сохраняет предсказания нейросети в CSV-файл для сабмита."""
    X = torch.tensor(preproc.transform(X_submit), dtype=torch.float32)

    predictions = _nn_predict(model, X, batch_size=32)
    predictions = np.expm1(predictions)

    submission = pd.DataFrame({"Id": X_submit.index, cfg.target_column: predictions})

    submission_path = os.path.join(cfg.data.submission_path, f"{submission_name}_submission.csv")
    submission.to_csv(submission_path, index=False)

    _log(f"Submission saved: {submission_path}", cfg.logging.console)
    return submission_path


def nn_train_epoch(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None = None,
) -> float:
    """Обучает нейросеть один цикл по всем батчам."""
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = loss_fn(outputs.squeeze(), batch_y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def nn_eval(
    model: nn.Module,
    loss_fn: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    methods: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Вычисляет значение функции потерь и выбранных метрик на валидации."""
    model.eval()

    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs.squeeze(), y_val)

        metric_to_score = {}
        for name, metric_cfg in methods.items():
            metric = hydra.utils.instantiate(metric_cfg)

            y_pred = val_outputs.squeeze().float()
            metric_score = metric(y_pred, y_val)

            if torch.is_tensor(metric_score):
                metric_score = metric_score.item()

            metric_to_score[name] = float(metric_score)

    return val_loss, metric_to_score


def nn_train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: DictConfig,
    logger: Any = None,
) -> dict[str, Any]:
    """Запускает полный цикл обучения нейросети с предобработкой и сохранением чекпоинта."""
    console = cfg.logging.console

    preproc = build_preprocessor(
        cfg=cfg,
        model_cfg=cfg.model.nn_model,
        is_scale=cfg.model.nn_model.is_scale,
        is_cat=cfg.model.nn_model.is_cat,
    )

    X_train_t = torch.tensor(preproc.fit_transform(X_train), dtype=torch.float32)
    X_val_t = torch.tensor(preproc.transform(X_val), dtype=torch.float32)

    if X_train_t.ndim == 1:
        X_train_t = X_train_t.unsqueeze(1)
    if X_val_t.ndim == 1:
        X_val_t = X_val_t.unsqueeze(1)

    n_features = X_train_t.shape[1]

    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Dropout(cfg.model.nn_model.dropout_rate),
        nn.BatchNorm1d(32),
        nn.Linear(32, 1),
    )

    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.model.nn_model.batch_size,
        shuffle=cfg.model.nn_model.shuffle,
    )

    optimizer = hydra.utils.instantiate(cfg.model.nn_model.optimizer, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.model.nn_model.loss_function)
    scheduler = hydra.utils.instantiate(cfg.model.nn_model.scheduler, optimizer=optimizer)
    epochs = cfg.model.nn_model.epochs

    best_loss = float("inf")
    patience = cfg.model.nn_model.patience
    patience_counter = 0

    for epoch in range(epochs):
        avg_train_loss = nn_train_epoch(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
        val_loss, metrics = nn_eval(
            model,
            loss_fn=loss_fn,
            X_val=X_val_t,
            y_val=y_val_t,
            methods=OmegaConf.to_container(cfg.model.nn_model.metrics, resolve=True),
        )

        if val_loss.item() < best_loss - cfg.model.nn_model.min_delta:
            best_loss = val_loss.item()
            patience_counter = 0

            model_name = model.__class__.__name__
            filename = model_filename(cfg, model_name, "states", -metrics["rmse"], extension="pt")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "val_rmse": metrics["rmse"],
            }

            torch.save(checkpoint, filename)

            if logger is not None:
                logger.log_pipeline(filename)
            _log(f"Model states saved as: {filename}", console)
        else:
            patience_counter += 1

        scheduler.step(val_loss)

        if patience_counter >= patience:
            _log(f"Early stopping triggered at epoch {epoch + 1}", console)
            break

        _log(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss.item():.4f}, "
            f"Val RMSE: {metrics['rmse']:.4f}",
            console,
        )

    return {
        "model": model,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "preproc": preproc,
        "path": filename,
    }


def add_nn_res(
    metric_to_score: dict[str, float],
    loss: float,
    tuning_time_sec: float,
    predict_time: float,
    latency_ms: float,
    model_cfg: DictConfig,
    path: str,
    results: list[dict[str, Any]] | None = None,
    log_file_path: str | None = None,
) -> dict[str, Any]:
    """Формирует словарь результата нейросетевого эксперимента и пишет его в JSONL."""
    experiment_data: dict[str, Any] = {
        "model": "NN_Model",
        "mse": metric_to_score.get("mse", None),
        "rmse": metric_to_score.get("rmse", None),
        "r2": metric_to_score.get("r2", None),
        "mae": metric_to_score.get("mae", None),
        "loss": loss,
        "params": OmegaConf.to_container(model_cfg, resolve=True),
        "tuning_time_sec": tuning_time_sec,
        "predict_time_sec": predict_time,
        "latency_ms_per_sample": latency_ms,
        "path": path,
    }

    if results is not None:
        results.append(experiment_data)

    if log_file_path:
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")

    return experiment_data


def nn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_submit: pd.DataFrame | None,
    cfg: DictConfig,
    logger: Any = None,
) -> None:
    """Оркестрирует обучение, оценку и сохранение результатов нейросетевой модели."""
    pipeline_output = run_method(
        obj=nn_train_pipeline,
        method_name="__call__",
        stage="nn_pipeline",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
        logger=logger,
    )

    pipeline_res = pipeline_output["result"]
    cycle_time = pipeline_output["nn_pipeline_time_sec"]
    model = pipeline_res["model"]
    X_val = pipeline_res["X_val"]
    y_val = pipeline_res["y_val"]
    preproc = pipeline_res["preproc"]
    path = pipeline_res["path"]

    loss_fn = hydra.utils.instantiate(cfg.model.nn_model.loss_function)

    eval_output = run_method(
        obj=nn_eval,
        method_name="__call__",
        stage="nn_predict",
        model=model,
        loss_fn=loss_fn,
        X_val=X_val,
        y_val=y_val,
        methods=OmegaConf.to_container(cfg.model.nn_model.metrics, resolve=True),
    )

    val_loss, metrics = eval_output["result"]
    predict_time = eval_output["nn_predict_time_sec"]
    tuning_time_sec = cycle_time - predict_time
    num_samples = X_val.shape[0]
    latency_ms_per_sample = (predict_time * 1000) / num_samples

    add_nn_res(
        metric_to_score=metrics,
        loss=val_loss.item(),
        tuning_time_sec=tuning_time_sec,
        predict_time=predict_time,
        latency_ms=latency_ms_per_sample,
        model_cfg=cfg.model.nn_model,
        path=path,
        log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"),
    )

    if X_submit is not None and preproc is not None:
        submission_path = save_nn_submission(
            model=model,
            X_submit=X_submit,
            cfg=cfg,
            preproc=preproc,
            submission_name="House_Prices_NN_Model",
        )
        _log(f"Submission saved: {submission_path}", cfg.logging.console)
