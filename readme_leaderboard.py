import json
import re

import pandas as pd

__all__ = [
    "build_leaderboard_table",
    "leaderboard_to_markdown",
    "load_leaderboard",
    "update_readme_leaderboard",
]

def load_leaderboard(log_file_path: str) -> pd.DataFrame:
    """Читает jsonl файл и строит таблицу лучших результатов по каждой модели."""
    records = [json.loads(line) for line in open(log_file_path, encoding="utf-8")]
    df = pd.DataFrame(records)
    return df

def build_leaderboard_table(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Создание таблицы для README."""
    best = (
        df.sort_values(metric, ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", metric, "tuning_time_sec", "latency_ms_per_sample"]]
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )
    return best

def leaderboard_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def update_readme_leaderboard(readme_path: str, leaderboard_md: str) -> None:
    """Находит теги в README и заменяет текст между ними на актуальную таблицу."""
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