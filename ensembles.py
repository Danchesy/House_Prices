import os
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
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


class PreTrainedStackingRegressor:
    def __init__(self, estimators, final_estimator):
        self.estimators = estimators
        self.final_estimator = final_estimator
        
    def _get_meta_features(self, X):
        meta_features = []
        for name, pipe in self.estimators:
            preds = pipe.predict_proba(X)[:, 1]
            meta_features.append(preds)
        return np.column_stack(meta_features)

    def fit(self, X, y):
        X_meta = self._get_meta_features(X)
        self.final_estimator.fit(X_meta, y)
        return self

    def predict(self, X):
        X_meta = self._get_meta_features(X)
        return self.final_estimator.predict(X_meta)

    def score(self, X, y):
        return root_mean_squared_error(y, self.predict(X))

    def get_params(self):
        """Возвращает параметры для совместимости с логированием."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators]
        }


class PreTrainedVotingRegressor:
    def __init__(self, estimators):
        self.estimators = estimators  # Список кортежей [(name, pipe), ...]

    def fit(self, X=None, y=None):
        # Заглушка для совместимости с API sklearn.
        # Берутся предобученные модели.
        return self

    def predict(self, X):
        all_votes = [pipe.predict(X) for _, pipe in self.estimators]
        return np.mean(all_votes, axis=0)

    def score(self, X, y):
        return root_mean_squared_error(y, self.predict(X))

    def get_params(self):
        """Возвращает параметры для совместимости с логированием."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators]
        }


def load_stacking_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None):
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())
    
    return PreTrainedStackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(max_iter=1000, random_state=42)
    )


def load_voting_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None):
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())
    
    return PreTrainedVotingRegressor(estimators=estimators)


def load_pipelines(leaderboard: pd.DataFrame, top_k: int | None = None):
    best_models = (
        leaderboard.sort_values(by="rmse", ascending=False)
            .groupby("model", as_index=False)
            .first()
    )

    if top_k is not None:
        best_models = best_models.head(top_k)

    pipeline_paths = [Path(path) for path in best_models['path'].to_list()]
    pipeline_names = best_models['model'].to_list()

    return {name: joblib.load(path) for path, name in zip(pipeline_paths, pipeline_names)}


def ensemble_return(
    model,
    metric_to_score,
    path,
    tuning_time: float | None = None,
    predict_time: float | None = None,
    n_samples: int | None = None,
) -> dict:
    result = {
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
            # Latency: 1 obj per ms
            latency_ms = (predict_time / n_samples) * 1000
            result["latency_ms_per_sample"] = round(latency_ms, 4)

    return result


def make_ensembles(X_train, X_val, y_train, y_val, X_submit, methods, cfg, logger=None):    
    console = cfg.logging.console
    log_file_path = os.path.join(cfg.data.results_dir, "experiments.jsonl")

    leaderboard_df = load_leaderboard(log_file_path)

    ensemble_configs = []
    for ens_cfg in cfg.model.ensemble.list:
        ensemble_configs.append({
            "suffix": ens_cfg.suffix,
            "factory": lambda ec=ens_cfg: hydra.utils.instantiate(ec.factory, leaderboard=leaderboard_df)
        })

    for ens in ensemble_configs:
        suffix = ens["suffix"]
        model = ens["factory"]()
        
        _log(f"\n {model.__class__.__name__}", console)

        train_output = run_method(
            obj=model,
            method_name="fit",
            stage='train',
            X=X_train,
            y=y_train,
        )

        pred_output = holdout_score(model, X_val, y_val, metric=cfg.tuning.metric)

        _log(f"Holdout {cfg.tuning.metric}: {pred_output['result']:.4f}", console)
        _log(f"Final pipeline's training: {train_output['train_time_sec']:.4f} s.", console)
        _log(
            f"Holdout predictions ({len(X_val)} lines): {pred_output['predict_time_sec']:.4f} s.",
            console,
        )

        y_pred_holdout = model.predict(X_val)
        metric_to_score = {}
        for name, metric_cfg in methods.items():
            metric_fn = hydra.utils.instantiate(metric_cfg)
            score = metric_fn(y_val, y_pred_holdout)
            metric_to_score[name] = float(score)
            _log(f"Holdout {name}: {score:.4f}", console)

        model_name = model.__class__.__name__
        filename = model_filename(cfg, model_name, f"ensemble_{suffix}", pred_output["result"])
        
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
