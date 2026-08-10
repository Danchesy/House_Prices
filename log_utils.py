import json
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "WandbLogger",
    "_log",
    "add_result",
]

class WandbLogger:
    """Логирование экспериментов в Weights & Biases."""

    def __init__(self, cfg: DictConfig) -> None:
        self.enabled = cfg.logging.wandb.enabled

        if not self.enabled:
            return

        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            tags=cfg.logging.wandb.tags,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log_experiment(self, experiment: dict[str, Any]) -> None:
        """Логирует результаты одного эксперимента."""
        if not self.enabled:
            return

        log_data = {
            "mse": experiment["mse"],
            "std": experiment["std"],
            "tuning_time_sec": experiment["tuning_time_sec"],
            "predict_time_sec": experiment["predict_time_sec"],
            "latency_ms_per_sample": experiment["latency_ms_per_sample"],
        }

        for key, value in experiment["params"].items():
            log_data[f"param/{key}"] = value

        wandb.log(log_data)

    def log_pipeline(self, model_path: str) -> None:
        """Сохраняет обученную модель."""
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name="best_model",
            type="model",
        )

        artifact.add_file(model_path)

        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Завершает текущий run."""
        if self.enabled:
            wandb.finish()


def add_result(
    output: dict[str, Any],
    results: list[dict[str]] | list[dict[Any]] | None = None,
    log_file_path: str | None = None,
) -> dict[str, Any]:
    """
    Добавляет результат в список и дописывает его в файл на диске.

    Args:
        output: Словарь-результат эксперимента
        results: Глобальный список для хранения истории в текущей сессии
        log_file_path: Путь к файлу на диске, куда будут дописываться логи

    Returns:
        Dict[str, Any]: Словарь с данными эксперимента
    """
    if "model" in output:
        model_name = type(output["model"]).__name__
    elif "pipeline" in output:
        model_name = type(output["pipeline"].named_steps["model"]).__name__
    else:
        model_name = "UnknownModel"

    raw_params = output.get("params", {})

    experiment_data = {
        "model": model_name,
        "mse": float(output.get("mse")),
        "rmse": float(output.get("rmse")),
        "r2": float(output.get("r2_score")),
        "mae": float(output.get("mae")),
        "std": float(output.get("std_score", 0.0)),
        "params": {str(k): str(v) for k, v in raw_params.items()},
        "tuning_time_sec": output.get("tuning_time_sec"),
        "predict_time_sec": output.get("predict_time_sec"),
        "latency_ms_per_sample": output.get("latency_ms_per_sample", 0.0),
        "path": output.get("path"),
    }

    if results is not None:
        results.append(experiment_data)

    # Дописываем в файл ('a' — append)
    # JSON Lines (один эксперимент — одна строчка в файле)
    if log_file_path:
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")

    return experiment_data


def _log(message: str, console: bool) -> None:
    if console:
        print(message)
