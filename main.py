import warnings
from functools import partial

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ensembles import make_ensembles
from log_utils import WandbLogger
from nn_model import nn_model
from readme_leaderboard import *
from tuning_params import *
from utils import data_loading, get_all_categorical_columns, set_seed

warnings.filterwarnings("ignore")

OPTUNA_MAP = {
    "linear_model": linreg_optuna_params,
    "knn": knn_optuna_params,
    "dt": dt_optuna_params,
    "rf": rf_optuna_params,
    "xgboost": xgb_optuna_params,
    "lightgbm": lgbm_optuna_params,
    "catboost": catboost_optuna_params,
}

META_KEYS = {
    "grid_params",
    "is_cat",
    "is_scale",
    "cat_features",
    "scoring",
    "refit",
    "Cs",
    "scaler",
    "encoder",
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Главная функция запуска экспериментов."""
    if cfg.logging.console:
        print("ЗАПУСК ЭКСПЕРИМЕНТОВ")

    set_seed(cfg.seed)
    logger = WandbLogger(cfg)

    X_train, X_val, y_train, y_val, X_submit = data_loading(cfg)

    if cfg.logging.console:
        print(
            f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_submit.shape}"
        )

    for model_name, model_cfg in cfg.model.items():
        if cfg.logging.console:
            print(f"\n{model_name.upper()}")

        if model_name == "nn_model":
            nn_model(X_train, np.log1p(y_train), X_val, np.log1p(y_val), X_submit, cfg, logger=logger)
            continue

        elif model_name == "ensemble":
            make_ensembles(X_train, X_val, np.log1p(y_train), np.log1p(y_val), X_submit, methods=cfg.tuning.metrics,
                    cfg=cfg,
                    logger=logger,
                )    
            continue

        clean_cfg = OmegaConf.create(
            {k: v for k, v in model_cfg.items() if k not in META_KEYS}
        )

        # cat_features = (
        #     list(OmegaConf.to_container(model_cfg.cat_features, resolve=True))
        #     if "cat_features" in model_cfg
        #     else None
        # )

        model = hydra.utils.instantiate(clean_cfg)

        grid_p = (
            OmegaConf.to_container(model_cfg.grid_params, resolve=True)
            if "grid_params" in model_cfg
            else {}
        )

        optuna_p = OPTUNA_MAP.get(model_name)

        if model_name == "linear_model":
            optuna_p = partial(linreg_optuna_params)

        is_scale = model_cfg.get("is_scale", False)
        is_cat = model_cfg.get("is_cat", True)        
        # is_scale = model_cfg.get("is_scale", cfg.preprocessing.scale_numeric)
        # is_cat = model_cfg.get("is_cat", cfg.preprocessing.encode_categorical)

        if cfg.tuning.enabled and grid_p and cfg.tuning.use_grid_search:
            grid_tuning(
                model,
                grid_p,
                X_train,
                y_train,
                X_val,
                y_val,
                cfg,
                model_cfg,
                methods=cfg.tuning.metrics,
                is_scale=is_scale,
                is_cat=is_cat,
                cat_features=get_all_categorical_columns(cfg),
                X_submit=X_submit,
                logger=logger,
            )

        if cfg.tuning.enabled and optuna_p and cfg.tuning.use_optuna:
            optuna_tuning(
                model,
                optuna_p,
                X_train,
                y_train,
                X_val,
                y_val,
                cfg,
                model_cfg,
                methods=cfg.tuning.metrics,
                n_trials=cfg.tuning.n_trials,
                is_scale=is_scale,
                is_cat=is_cat,
                cat_features=get_all_categorical_columns(cfg),
                X_submit=X_submit,
                logger=logger,
            )

    table = (
        load_leaderboard("results/experiments.jsonl")
        .pipe(build_leaderboard_table)
        .pipe(leaderboard_to_markdown)
    )

    update_readme_leaderboard("README.md", table)

    logger.finish()
    if cfg.logging.console:
        print("\nГотово")


if __name__ == "__main__":
    main()
