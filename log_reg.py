from log_utils import add_result
from typing import Any

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from utils import pipeline_return


def logistic_kfold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    logger: Any = None,
) -> dict[str, Any]:
    """
    Выполняет кросс-валидацию логистической регрессии с предобработкой данных.

    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        X_test: Тестовые признаки (для логирования)
        y_test: Тестовая целевая переменная (для логирования)
        cfg: Конфигурация Hydra
        logger: Объект логгера

    Returns:
        Dict[str, Any]: Словарь с результатами
    """
    params = cfg.model.linear_model
    training = cfg.training
    determinism = cfg

    penalty = params.penalty
    C = params.C
    l1_ratio = params.l1_ratio
    solver = params.solver
    tol = params.tol
    fit_intercept = params.fit_intercept
    random_state = determinism.seed
    max_iter = params.max_iter
    verbose = params.verbose
    n_splits = training.cv_folds
    shuffle = training.shuffle
    scoring = params.get("scoring", "accuracy")

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        tol=tol,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    cfg,
                    params,
                    is_scale=params.is_scale,
                    is_cat=params.is_cat,
                ),
            ),
            ("model", model),
        ]
    )

    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=scoring)

    if cfg.logging.console:
        for fold, acc in enumerate(scores, start=1):
            print(f"fold-{fold} accuracy: {acc}")

    pipeline.fit(X_train, y_train)

    model = pipeline.named_steps["model"]

    if cfg.logging.console:
        print(f"{model} mean accuracy: {np.mean(scores)}")
        print(f"Test accuracy: {pipeline.score(X_test, y_test)}")

    res = pipeline_return(pipeline, scores)
    experiment = add_result(res)

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model.__class__.__name__
    filename = (
        f"{cfg.data.models_dir}/{model_name}_log_reg_cv_{np.mean(scores):.4f}.pkl"
    )
    joblib.dump(pipeline, filename)

    if logger is not None:
        logger.log_pipeline(filename)

    print(f"Pipeline saved as: {filename}")

    return {
        "model": model,
        "fold_scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "params": model.get_params(),
    }


def log_reg_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    logger: Any = None,
) -> dict[str, Any]:
    """
    Выполняет поиск гиперпараметров логистической регрессии с помощью встроенной кросс-валидации.

    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        X_test: Тестовые признаки
        y_test: Тестовая целевая переменная
        cfg: Конфигурация Hydra
        logger: Объект логгера

    Returns:
        Dict[str, Any]: Словарь с результатами
    """
    params = cfg.model.linear_model
    training = cfg.training
    determinism = cfg

    model_cv = LogisticRegressionCV(
        tol=params.tol,
        fit_intercept=params.fit_intercept,
        random_state=determinism.seed,
        max_iter=params.max_iter,
        solver=params.solver,
        cv=training.cv_folds,
        refit=params.get("refit", True),
        Cs=params.get("Cs", 10),
        verbose=params.verbose,
        scoring=params.get("scoring", "accuracy"),
    )

    pipeline_cv = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    cfg,
                    params,
                    is_scale=params.is_scale,
                    is_cat=params.is_cat,
                ),
            ),
            ("model", model_cv),
        ]
    )

    pipeline_cv.fit(X_train, y_train)

    best_model = pipeline_cv.named_steps["model"]
    best_C = best_model.C_.item()

    scores = best_model.scores_

    test_acc = pipeline_cv.score(X_test, y_test)

    label = next(iter(scores.keys()))
    mean_cv_per_c = np.mean(scores[label], axis=0)
    best_cv_score = np.max(mean_cv_per_c).item()

    print(f"best C: {best_C}")
    print(f"mean acc for best C: {best_cv_score}")
    print(f"test accuracy: {test_acc}")

    res = pipeline_return(pipeline_cv, best_cv_score)
    experiment = add_result(res)

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model_cv.__class__.__name__
    filename = f"{cfg.data.models_dir}/{model_name}_log_reg_cv_{best_cv_score:.4f}.pkl"
    joblib.dump(pipeline_cv, filename)

    if logger is not None:
        logger.log_pipeline(filename)

    print(f"Pipeline saved as: {filename}")

    return {
        "model": best_model,
        "best_C": best_C,
        "best_cv_score": best_cv_score,
        "test_score": test_acc,
    }
