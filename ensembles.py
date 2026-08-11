import os
from pathlib import Path
from typing import Any

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from log_utils import _log, add_result
from readme_leaderboard import load_leaderboard
from utils import (
    holdout_score,
    model_filename,
    run_method,
    save_submission,
    submission_output_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_pipeline_path(path_value: str | os.PathLike[str] | None) -> Path | None:
    """Пытается найти путь к сохранённой модели по относительному или именованному пути."""
    if path_value is None:
        return None

    raw_path = Path(str(path_value))
    if raw_path.is_absolute():
        return raw_path if raw_path.exists() else None

    candidates = [raw_path, PROJECT_ROOT / raw_path, Path.cwd() / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if raw_path.name:
        for base_dir in [
            PROJECT_ROOT / "models",
            Path.cwd() / "models",
            PROJECT_ROOT,
            Path.cwd(),
        ]:
            candidate = base_dir / raw_path.name
            if candidate.exists():
                return candidate.resolve()

    return None


class PreTrainedStackingRegressor:
    """Регрессор-стекер, который использует уже обученные модели как базовые предикторы."""

    def __init__(self, estimators: list[tuple[str, Any]], final_estimator: Any) -> None:
        """Сохраняет базовые модели и финальный мета-алгоритм."""
        self.estimators = estimators
        self.final_estimator = final_estimator

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Получает мета-признаки из предсказаний базовых моделей."""
        meta_features: list[np.ndarray] = []
        for _, pipe in self.estimators:
            preds = pipe.predict(X)
            meta_features.append(preds)
        return np.column_stack(meta_features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreTrainedStackingRegressor":
        """Обучает финальный мета-регрессор на предсказаниях базовых моделей."""
        X_meta = self._get_meta_features(X)
        self.final_estimator.fit(X_meta, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Делает предсказание через мета-регрессор."""
        X_meta = self._get_meta_features(X)
        return self.final_estimator.predict(X_meta)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Возвращает RMSE для оценки качества ансамбля."""
        return root_mean_squared_error(y, self.predict(X))

    def get_params(self) -> dict[str, Any]:
        """Возвращает параметры для совместимости с логированием."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators],
        }


class PreTrainedVotingRegressor:
    """Регрессор-голосование, который агрегирует предсказания обученных моделей."""

    def __init__(self, estimators: list[tuple[str, Any]]) -> None:
        """Сохраняет список базовых моделей для агрегации."""
        self.estimators = estimators

    def fit(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> "PreTrainedVotingRegressor":
        """Поддерживает совместимость с API sklearn, не обучая модели заново."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Среднее предсказание всех базовых моделей."""
        all_votes = [pipe.predict(X) for _, pipe in self.estimators]
        return np.mean(all_votes, axis=0)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Возвращает RMSE для оценки качества ансамбля."""
        return root_mean_squared_error(y, self.predict(X))

    def get_params(self) -> dict[str, Any]:
        """Возвращает параметры для совместимости с логированием."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators],
        }


def load_stacking_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None) -> PreTrainedStackingRegressor:
    """Создаёт стекер из лучших моделей из leaderboard."""
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())

    return PreTrainedStackingRegressor(estimators=estimators, final_estimator=LinearRegression())


def load_voting_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None) -> PreTrainedVotingRegressor:
    """Создаёт voting-регрессор из лучших моделей из leaderboard."""
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())

    return PreTrainedVotingRegressor(estimators=estimators)


def load_pipelines(leaderboard: pd.DataFrame, top_k: int | None = None) -> dict[str, Any]:
    """Загружает сохранённые пайплайны по путям из leaderboard."""
    best_models = (
        leaderboard.sort_values(by="rmse", ascending=True)
        .groupby("model", as_index=False)
        .first()
    )

    if top_k is not None:
        best_models = best_models.head(top_k)

    pipelines = {}
    for _, row in best_models.iterrows():
        resolved_path = _resolve_pipeline_path(row["path"])
        if resolved_path is None:
            continue
        pipelines[row["model"]] = joblib.load(resolved_path)

    if not pipelines:
        raise FileNotFoundError("No saved model artifacts were found for the ensemble leaderboard entries.")

    return pipelines


def ensemble_return(
    model: Any,
    metric_to_score: dict[str, float],
    path: str,
    tuning_time: float | None = None,
    predict_time: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Формирует словарь с метаданными ансамбля для логирования."""
    result: dict[str, Any] = {
        "model": model,
        "mse": metric_to_score.get("mse", None),
        "rmse": metric_to_score.get("rmse", None),
        "r2": metric_to_score.get("r2", None),
        "mae": metric_to_score.get("mae", None),
        "params": model.get_params(),
        "path": path,
    }

    if tuning_time is not None:
        result["tuning_time_sec"] = round(tuning_time, 2)

    if predict_time is not None:
        result["predict_time_sec"] = round(predict_time, 4)
        if n_samples and n_samples > 0:
            latency_ms = (predict_time / n_samples) * 1000
            result["latency_ms_per_sample"] = round(latency_ms, 4)

    return result


def make_ensembles(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_submit: pd.DataFrame | None,
    methods: dict[str, Any],
    cfg: DictConfig,
    logger: Any = None,
) -> None:
    """Создаёт и оценивает ансамбли из лучших моделей по leaderboard."""
    console = cfg.logging.console
    log_file_path = os.path.join(cfg.data.results_dir, "experiments.jsonl")

    leaderboard_df = load_leaderboard(log_file_path)

    ensemble_configs = []
    for ens_cfg in cfg.model.ensemble.list:
        ensemble_configs.append(
            {
                "suffix": ens_cfg.suffix,
                "factory": lambda ec=ens_cfg: hydra.utils.instantiate(ec.factory, leaderboard=leaderboard_df),
            }
        )

    for ens in ensemble_configs:
        suffix = ens["suffix"]
        model = ens["factory"]()

        _log(f"\n {model.__class__.__name__}", console)

        train_output = run_method(obj=model, method_name="fit", stage="train", X=X_train, y=y_train)

        pred_output = holdout_score(model, X_val, y_val, metric=cfg.tuning.metric)

        _log(f"Holdout {cfg.tuning.metric}: {pred_output['result']:.4f}", console)
        _log(f"Final pipeline's training: {train_output['train_time_sec']:.4f} s.", console)
        _log(f"Holdout predictions ({len(X_val)} lines): {pred_output['predict_time_sec']:.4f} s.", console)

        y_pred_holdout = model.predict(X_val)
        metric_to_score = {}
        for name, metric_cfg in methods.items():
            metric_fn = hydra.utils.instantiate(metric_cfg)
            score = metric_fn(y_val, y_pred_holdout)
            metric_to_score[name] = float(score)
            _log(f"Holdout {name}: {score:.4f}", console)

        model_name = model.__class__.__name__
        filename = model_filename(cfg, model_name, f"ensemble_{suffix}", -pred_output["result"])

        res = ensemble_return(
            model=model,
            metric_to_score=metric_to_score,
            path=filename,
            tuning_time=train_output["train_time_sec"],
            predict_time=pred_output["predict_time_sec"],
            n_samples=len(X_val),
        )

        latency_ms_per_sample = (pred_output["predict_time_sec"] * 1000) / X_val.shape[0]
        res.update({"latency_ms_per_sample": latency_ms_per_sample})

        experiment = add_result(res, log_file_path=log_file_path)

        if logger is not None:
            logger.log_experiment(experiment)

        if cfg.logging.save_model:
            joblib.dump(model, filename)
            if logger is not None:
                logger.log_pipeline(filename)
            _log(f"Pipeline saved as: {filename}", console)

        if cfg.logging.save_predictions and X_submit is not None:
            submit_path = submission_output_path(cfg, model_name)
            save_submission(model, X_submit, submit_path)
