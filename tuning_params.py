import os
from collections.abc import Callable
from typing import Any

import hydra
import joblib
import numpy as np
import optuna
import pandas as pd
from log_utils import _log, add_result
from omegaconf import DictConfig
from preprocessing import build_preprocessor, pipeline_fit_params
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from utils import (
    ensure_dirs,
    holdout_score,
    model_filename,
    pipeline_return,
    run_method,
    save_submission,
    submission_output_path,
)

__all__ = [
    "catboost_optuna_params",
    "dt_optuna_params",
    "grid_tuning",
    "knn_optuna_params",
    "lgbm_optuna_params",
    "linreg_optuna_params",
    "optuna_tuning",
    "rf_optuna_params",
    "xgb_optuna_params",
]


def grid_tuning(
    model: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    model_cfg: DictConfig,
    methods: dict[str, Any],
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
    X_submit: pd.DataFrame | None = None,
    logger: Any = None,
) -> GridSearchCV:
    """Выполняет GridSearchCV для подбора гиперпараметров и логирования результатов.

    Args:
        model: sklearn-совместимый классификатор.
        params: сетка гиперпараметров с префиксом ``model__``.
        X_train, y_train: обучающая выборка.
        X_test, y_test: валидационная выборка для итоговой оценки.
        cfg: Конфигурационный объект Hydra (DictConfig), содержит секции logging, tuning, training, data.
        model_cfg: Конфигурация модели (DictConfig), передаётся в build_preprocessor.
        methods: Словарь метрик для дополнительной оценки, где ключ — имя метрики, значение — конфиг метрики.
        is_scale: передаётся в :func:`preprocessor`.
        is_cat: передаётся в :func:`preprocessor`.
        cat_features: Список имён категориальных признаков (передаётся в pipeline_fit_params).
        X_submit: Данные для генерации файла сабмишена (опционально).
        logger: объект с методом ``log_experiment`` / ``log_pipeline``.

    Returns:
        Обученный ``GridSearchCV`` с лучшим estimator'ом в ``best_estimator_``.
    """

    ensure_dirs(cfg)
    console = cfg.logging.console
    metric = cfg.tuning.metric
    cv_folds = cfg.training.cv_folds
    n_jobs = cfg.training.n_jobs

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(cfg, model_cfg, is_scale=is_scale, is_cat=is_cat)),
            ("model", model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=metric,
        refit=True,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=1 if console else 0,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
    )

    train_output = run_method(
        obj=grid_search,
        method_name="fit",
        stage="train",
        X=X_train,
        y=y_train,
        **pipeline_fit_params(model, cat_features),
    )

    _log(f"Best parameters: {grid_search.best_params_}", console)
    _log(f"Best CV {metric}: {grid_search.best_score_:.4f}", console)
    _log(f"GridSearch training time: {train_output['train_time_sec']:.2f} s.", console)

    best_pipeline = grid_search.best_estimator_
    best_idx = grid_search.best_index_

    cv_scores = np.array(
        [
            grid_search.cv_results_[f"split{i}_test_score"][best_idx]
            for i in range(cv_folds)
        ],
        dtype=float,
    )

    pred_output = holdout_score(pipeline=best_pipeline, X=X_test, y=y_test, metric=metric)

    _log(f"Holdout {metric}: {pred_output['result']:.4f}", console)
    _log(f"Holdout predict ({len(X_test)} lines): {pred_output['result']:.4f} s.", console)
    _log(f"Latency: {(pred_output['predict_time_sec'] / len(X_test)) * 1000:.4f} ms", console)

    y_pred_holdout = best_pipeline.predict(X_test)

    metric_to_score = {}
    for name, metric_cfg in methods.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        score = metric_fn(y_test, y_pred_holdout)
        metric_to_score[name] = float(score)
        _log(f"Holdout {name}: {score:.4f}", console)

    res = pipeline_return(
        pipeline=best_pipeline,
        cv_scores=cv_scores,
        tuning_time=train_output["train_time_sec"],
        predict_time=pred_output["predict_time_sec"],
        n_samples=len(X_test),
    )

    model_name = model.__class__.__name__
    filename = model_filename(cfg, model_name, "grid", -grid_search.best_score_)

    res.update(metric_to_score)
    res.update({"path": filename})

    experiment = add_result(res, log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"))

    if logger is not None:
        logger.log_experiment(experiment)

    if cfg.logging.save_model:
        joblib.dump(best_pipeline, filename)
        if logger is not None:
            logger.log_pipeline(filename)
        _log(f"Pipeline saved as: {filename}", console)

    if cfg.logging.save_predictions and X_submit is not None:
        submit_path = submission_output_path(cfg, model_name)
        save_submission(best_pipeline, X_submit, submit_path)

    return grid_search


def optuna_tuning(
    model: Any,
    params_fn: Callable[[optuna.Trial], dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    model_cfg: DictConfig,
    methods: dict[str, Any],
    n_trials: int = 20,
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
    X_submit: pd.DataFrame | None = None,
    logger: Any = None,
) -> optuna.Study:
    """Выполняет Optuna-оптимизацию гиперпараметров и обучает финальный пайплайн.

    Args:
        model: Базовый sklearn-совместимый классификатор/регрессор (клонируется для каждого trial).
        params_fn: Функция, принимающая ``optuna.Trial`` и возвращающая словарь с гиперпараметрами для модели.
        X_train: Обучающие признаки (pd.DataFrame).
        y_train: Обучающие целевые значения (pd.Series).
        X_test: Валидационные признаки для итоговой оценки (pd.DataFrame).
        y_test: Валидационные целевые значения для итоговой оценки (pd.Series).
        cfg: Конфигурационный объект Hydra (DictConfig), содержит секции logging, tuning, training, data.
        model_cfg: Конфигурация модели (DictConfig), передаётся в build_preprocessor.
        methods: Словарь метрик для дополнительной оценки, где ключ — имя метрики, значение — конфиг метрики.
        n_trials: Количество испытаний (итераций оптимизации) Optuna.
        is_scale: Флаг, указывающий, нужно ли масштабировать признаки в препроцессоре.
        is_cat: Флаг, указывающий, нужно ли обрабатывать категориальные признаки в препроцессоре.
        cat_features: Список имён категориальных признаков (передаётся в pipeline_fit_params).
        X_submit: Данные для генерации файла сабмишена (опционально).
        logger: Объект логгера с методами ``log_experiment`` и ``log_pipeline`` (опционально).

    Returns:
        Завершённый ``optuna.Study`` с атрибутами ``best_params`` и ``best_value``.
    """

    ensure_dirs(cfg)
    console = cfg.logging.console
    metric = cfg.tuning.metric
    cv_folds = cfg.training.cv_folds
    direction = cfg.tuning.direction
    timeout = cfg.tuning.timeout
    n_jobs = cfg.training.n_jobs

    cv_params = pipeline_fit_params(model, cat_features)

    def objective(trial: optuna.Trial) -> float:
        """Оценивает один trial через кросс-валидацию."""
        params = params_fn(trial)

        current_model = clone(model)
        current_model.set_params(**params)

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(cfg, model_cfg, is_scale=is_scale, is_cat=is_cat)),
                ("model", current_model),
            ]
        )

        score = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=metric,
            n_jobs=n_jobs,
            params=cv_params,
        ).mean()
        return score

    study = optuna.create_study(direction=direction)
    optimizer_output = run_method(
        obj=study,
        method_name="optimize",
        stage="optuna",
        func=objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    best_params = study.best_trial.user_attrs.get("sklearn_params", study.best_params)

    _log(f"Best CV {metric}: {study.best_value:.4f}", console)
    _log(f"Best parameters: {best_params}", console)
    _log(f"Optuna optimization time ({n_trials} trials): {optimizer_output['optuna_time_sec']:.2f} s.", console)
    _log(f"Mean time per trial: {optimizer_output['optuna_time_sec'] / max(n_trials, 1):.2f} s.", console)

    best_model = clone(model)
    best_model.set_params(**best_params)

    final_pipeline = Pipeline(
        [("preprocessor", build_preprocessor(cfg, model_cfg, is_scale, is_cat)), ("model", best_model)]
    )

    cv_scores = cross_val_score(
        final_pipeline,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=metric,
        n_jobs=n_jobs,
        params=cv_params,
    )

    train_output = run_method(
        obj=final_pipeline,
        method_name="fit",
        stage="train",
        X=X_train,
        y=y_train,
        **pipeline_fit_params(model, cat_features),
    )

    pred_output = holdout_score(final_pipeline, X_test, y_test, metric)

    _log(f"Holdout {metric}: {pred_output['result']:.4f}", console)
    _log(f"Final pipeline's training: {train_output['train_time_sec']:.4f} s.", console)
    _log(f"Holdout predictions ({len(X_test)} lines): {pred_output['predict_time_sec']:.4f} s.", console)

    y_pred_holdout = final_pipeline.predict(X_test)

    metric_to_score = {}
    for name, metric_cfg in methods.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        score = metric_fn(y_test, y_pred_holdout)
        metric_to_score[name] = float(score)
        _log(f"Holdout {name}: {score:.4f}", console)

    res = pipeline_return(
        final_pipeline,
        cv_scores=cv_scores,
        tuning_time=optimizer_output["optuna_time_sec"],
        predict_time=pred_output["predict_time_sec"],
        n_samples=len(X_test),
    )

    model_name = model.__class__.__name__
    filename = model_filename(cfg, model_name, "optuna", -study.best_value)

    res.update(metric_to_score)
    res.update({"path": filename})

    experiment = add_result(res, log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"))

    if logger is not None:
        logger.log_experiment(experiment)

    if cfg.logging.save_model:
        joblib.dump(final_pipeline, filename)
        if logger is not None:
            logger.log_pipeline(filename)
        _log(f"Pipeline saved as: {filename}", console)

    if cfg.logging.save_predictions and X_submit is not None:
        submit_path = submission_output_path(cfg, model_name)
        save_submission(final_pipeline, X_submit, submit_path)

    return study


def linreg_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для линейной регрессии."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }


def knn_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для KNN Regressor."""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 21, step=2),
        "weights": trial.suggest_categorical("weights", ["distance", "uniform"]),
        "leaf_size": trial.suggest_categorical("leaf_size", [20, 30, 50]),
        "metric": trial.suggest_categorical("metric", ["minkowski", "manhattan", "euclidean"]),
    }


def dt_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для DecisionTreeRegressor."""
    return {
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.8]),
    }


def rf_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для RandomForestRegressor."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 5, 10]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.8]),
    }


def xgb_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для XGBoost Regressor."""
    return {
        # "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
    }


def lgbm_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для LightGBM Regressor."""
    return {
        # "objective": "regression",
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
    }


def catboost_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Возвращает пространство поиска гиперпараметров для CatBoost Regressor."""
    return {
        "loss_function": "RMSE",
        "iterations": trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
