import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

__all__ = [
    "build_leaderboard_table",
    "leaderboard_to_markdown",
    "load_leaderboard",
    "update_readme_leaderboard",
]


def load_leaderboard(log_file_path: str | Path) -> pd.DataFrame:
    """Читает JSONL-файл и строит таблицу лучших результатов по каждой модели."""
    path = Path(log_file_path)
    if not path.is_absolute():
        for candidate in [PROJECT_ROOT / path, Path.cwd() / path]:
            if candidate.exists():
                path = candidate
                break

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    return pd.DataFrame(records)


def build_leaderboard_table(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Создаёт таблицу лидеров для README на основе логов экспериментов."""
    if df.empty:
        return pd.DataFrame(columns=["model", "metric", "tuning_time_sec", "latency_ms_per_sample"])

    preferred_metrics = [
        metric,
        "rmse",
        "mse",
        "mae",
        "r2",
        "accuracy",
        "f1",
        "precision",
        "recall",
    ]
    resolved_metric = next((name for name in preferred_metrics if name in df.columns), None)

    if resolved_metric is None:
        numeric_columns = [
            col for col in df.select_dtypes(include="number").columns if col not in {"index", "level_0"}
        ]
        resolved_metric = numeric_columns[0] if numeric_columns else "metric"

    sort_ascending = resolved_metric in {"rmse", "mse", "mae", "loss"}

    best = (
        df.sort_values(resolved_metric, ascending=sort_ascending)
        .groupby("model", as_index=False)
        .first()
    )

    columns = ["model", resolved_metric]
    for extra_col in ["tuning_time_sec", "latency_ms_per_sample"]:
        if extra_col in best.columns:
            columns.append(extra_col)

    best = best[columns].sort_values(resolved_metric, ascending=sort_ascending).reset_index(drop=True)
    best = best.rename(columns={resolved_metric: "metric"})
    best.columns = ["model", "metric", *[col for col in best.columns[2:] if col != "metric"]]
    return best


def leaderboard_to_markdown(df: pd.DataFrame) -> str:
    """Преобразует DataFrame с лидерами в Markdown-таблицу."""
    return df.to_markdown(index=False)


def update_readme_leaderboard(readme_path: str | Path, leaderboard_md: str) -> None:
    """Заменяет блок таблицы в README на актуальный Markdown."""
    with open(readme_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    pattern = r"(<!-- leaderboard_start -->).*?(<!-- leaderboard_end -->)"
    replacement = f"\\1\n{leaderboard_md}\n\\2"

    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)

    if count == 0:
        print("Ошибка: Теги <!-- leaderboard_start --> не найдены в README!")
        return

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Таблица в README.md успешно обновлена!")
