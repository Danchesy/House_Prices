import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

__all__ = [
    "data_loading",
    "ensure_dirs",
    "generate_submission",
    "get_all_categorical_columns",
    "holdout_score",
    "model_filename",
    "pipeline_return",
    "plot_feature_importance",
    "run_method",
    "save_submission",
    "set_seed",
    "submission_output_path",
    "time_and_score",
]

def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для всех используемых библиотек (Python, NumPy, PyTorch)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def data_loading(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Загружает и разделяет данные."""
    train = pd.read_csv(cfg.data.train_path)
    test = pd.read_csv(cfg.data.test_path)

    train.set_index("Id", inplace=True)
    test.set_index("Id", inplace=True)

    X = train.drop(columns=cfg.target_column)
    y = train[cfg.target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        shuffle=cfg.training.shuffle,
    )

    return X_train, X_val, y_train, y_val, test


def pipeline_return(
    pipeline: Pipeline,
    cv_scores: list[float] | np.ndarray,
    tuning_time: float | None = None,
    predict_time: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Формирует словарь с результатами пайплайна, включая метрики времени.

    Args:
        pipeline: Обученный пайплайн cv_scores: Массив оценок кросс-валидации или
        среднее значение tuning_time: Время подбора параметров (GridSearch /
        Optuna) в секундах predict_time: Общее время предсказания на тестовой
        выборке в секундах n_samples: Количество объектов в тестовой выборке для
        расчета latency

    Returns:
        Dict[str, Any]: Словарь с результатами и временными метриками
    """
    result = {
        "pipeline": pipeline,
        "mean_score": np.mean(cv_scores),
        "std_score": np.std(cv_scores),
        "params": pipeline.named_steps["model"].get_params(),
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


def save_submission(
    pipeline: Pipeline,
    X_submit: pd.DataFrame,
    submission_path: str,
) -> None:
    """Сохраняет предсказания в CSV для Kaggle."""
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    preds = pipeline.predict(X_submit)
    submission = pd.DataFrame(
        {"Id": X_submit.index, "SalePrice": preds.astype(float)}
    )
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved: {submission_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """
    Визуализирует важность признаков для моделей с атрибутом feature_importances_.

    Args:
        model: Обученная модель с атрибутом feature_importances_
        feature_names: Список названий признаков
        top_n: Количество наиболее важных признаков для отображения
        save_path: Путь для сохранения графика (если None — только show)
    """
    importances = model.feature_importances_

    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    fi_df = fi_df.head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=fi_df,
        palette="viridis",
        legend=False,
    )

    plt.title(
        f"Top {top_n} Feature Importances ({model.__class__.__name__})",
        fontsize=14,
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importance plot saved: {save_path}")
    plt.show()


def generate_submission(
    cfg: DictConfig,
    pipeline_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Генерирует файл сабмита из сохранённого пайплайна."""
    pipeline_path = pipeline_path or f"{cfg.data.models_dir}/full_pipeline.pkl"
    output_path = output_path or os.path.join(cfg.data.results_dir, "submission.csv")

    pipeline = joblib.load(pipeline_path)
    *_, test = data_loading(cfg)
    save_submission(pipeline, test, output_path)

def ensure_dirs(cfg: DictConfig) -> None:
    os.makedirs(cfg.data.models_dir, exist_ok=True)
    os.makedirs(cfg.data.results_dir, exist_ok=True)
    # os.makedirs(cfg.data.reports_dir, exist_ok=True)


def model_filename(
    cfg: DictConfig,
    model_name: str,
    method: str,
    score: float,
    extension: str = 'pkl',
) -> str:
    models_dir = Path(cfg.data.models_dir)
    
    prefix = f"{cfg.experiment_name}_" if cfg.get("experiment_name") else ""
    filename = f"{prefix}{model_name}_{method}_{score:.4f}.{extension}"
    
    return (models_dir / filename).as_posix()


def time_and_score(stage='train'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            final_stage = kwargs.pop('stage', stage)

            start_train = time.time()
            result = func(*args, **kwargs)
            timer = time.time() - start_train

            return {
                "result": result,
                f"{final_stage}_time_sec": timer
            }
        return wrapper
    return decorator


@time_and_score(stage='predict')
def holdout_score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
    scorer = get_scorer(metric)
    return float(scorer(pipeline, X, y))


@time_and_score()
def run_method(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name) if obj is not None else globals()[method_name]
    res = method(*args, **kwargs)

    return res


def submission_output_path(cfg: DictConfig, model_name: str) -> str:
    submit_dir = os.path.dirname(cfg.data.submission_path) or cfg.data.results_dir
    return os.path.join(submit_dir, f"{model_name}_submission.csv")

def get_all_categorical_columns(cfg: DictConfig) -> list[str]:
    """Объединяет все категориальные колонки из конфига."""
    cat_cols = []
    
    cat_keys = ["ohe_cols", "ordinal_cols", "quantile_cols"]
    
    for key in cat_keys:
        if key in cfg.preprocessing:
            cols = OmegaConf.to_container(cfg.preprocessing[key], resolve=True)
            if cols:
                cat_cols.extend(cols)
    
    return cat_cols