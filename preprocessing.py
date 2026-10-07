import inspect
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


class QuantileEncoderWrapper(BaseEstimator, TransformerMixin):
    """Совместимая обёртка над category_encoders.QuantileEncoder для sklearn Pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        """Сохраняет параметры и инициализирует внутренний encoder."""
        self.kwargs = kwargs
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "QuantileEncoderWrapper":
        """Обучает encoder на признаках и целевой переменной."""
        from category_encoders import QuantileEncoder

        self.encoder_ = QuantileEncoder(**self.kwargs)
        if y is None:
            y = pd.Series(0, index=X.index)
        self.encoder_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Преобразует данные с помощью обученного encoder."""
        check_is_fitted(self, "encoder_")
        if y is None:
            y = pd.Series(0, index=X.index)
        return self.encoder_.transform(X, y)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params: Any) -> pd.DataFrame:
        """Выполняет fit и transform за один проход."""
        self.fit(X, y)
        return self.transform(X, y)


__all__ = [
    "ORDINAL_CATEGORIES",
    "FeatureEngineer",
    "build_preprocessor",
    "pipeline_fit_params",
    "preprocessor",
]

NA_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

STANDARD_QUAL = ["Po", "Fa", "TA", "Gd", "Ex"]
STANDARD_QUAL_WITH_NA = ["NA", "Po", "Fa", "TA", "Gd", "Ex"]

ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "ExterQual": STANDARD_QUAL,
    "ExterCond": STANDARD_QUAL,
    "HeatingQC": STANDARD_QUAL,
    "KitchenQual": STANDARD_QUAL,
    "BsmtQual": STANDARD_QUAL_WITH_NA,
    "BsmtCond": STANDARD_QUAL_WITH_NA,
    "FireplaceQu": STANDARD_QUAL_WITH_NA,
    "GarageQual": STANDARD_QUAL_WITH_NA,
    "GarageCond": STANDARD_QUAL_WITH_NA,
    "PoolQC": STANDARD_QUAL_WITH_NA,
    "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
    "Fence": ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
}

CAT_COLUMN_KEYS: dict[str, str] = {
    "ohe_categorical": "ohe_cols",
    "quantile_categorical": "quantile_cols",
    "ordinal_categorical": "ordinal_cols",
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Выполняет базовую инженерию признаков для датасета House Prices."""

    def __init__(self, na_cols: list[str] | None = None) -> None:
        """Инициализирует набор колонок с пропусками и внутренние статистики."""
        self.na_cols = na_cols if na_cols is not None else NA_COLS
        self.electrical_mode_: str = "SBrkr"
        self.mas_mode_: str = "None"
        self.shape_to_frontage_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        """Считает статистики по данным для последующего заполнения пропусков."""
        X = X.copy()

        self.electrical_mode_ = X["Electrical"].mode().iloc[0]

        shape_to_area = X.groupby("LotShape")["LotArea"].median()
        self.shape_to_frontage_ = {
            shape: float(np.sqrt(area)) for shape, area in shape_to_area.items()
        }

        mas_mode = X["MasVnrType"].mode()
        self.mas_mode_ = mas_mode.iloc[0] if len(mas_mode) else "None"

        self.num_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.num_medians_ = X[self.num_cols_].median().to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Заполняет пропуски и подготавливает признаки для кодирования."""
        X = X.copy()

        for col in self.na_cols:
            if col in X.columns:
                X[col] = X[col].fillna("NA")

        X["Electrical"] = X["Electrical"].fillna(self.electrical_mode_)
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)

        for shape, frontage in self.shape_to_frontage_.items():
            mask = X["LotFrontage"].isna() & (X["LotShape"] == shape)
            X.loc[mask, "LotFrontage"] = frontage

        mask = (X["MasVnrArea"] > 0.0) & X["MasVnrType"].isna()
        X.loc[mask, "MasVnrType"] = self.mas_mode_
        X["MasVnrType"] = X["MasVnrType"].fillna("None")
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0.0)

        for col in self.num_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(self.num_medians_[col])

        cat_cols = X.select_dtypes(include=["object"]).columns
        X[cat_cols] = X[cat_cols].fillna("None")

        return X


def _resolve_columns(cfg: DictConfig, key: str) -> list[str]:
    """Возвращает список колонок из конфигурации preprocessing по заданному ключу."""
    return list(OmegaConf.to_container(cfg.preprocessing[key], resolve=True))


def _instantiate_ordinal_encoder(factory_cfg: DictConfig, columns: list[str]) -> Any:
    """Создаёт OrdinalEncoder с категориями, соответствующими колонкам."""
    cfg_dict = OmegaConf.to_container(factory_cfg, resolve=True)
    cfg_dict.pop("_target_", None)
    cfg_dict["categories"] = [list(ORDINAL_CATEGORIES[col]) for col in columns]
    cfg_dict["dtype"] = np.int64
    return OrdinalEncoder(**cfg_dict)


def _instantiate_quantile_encoder(factory_cfg: DictConfig) -> Any:
    """Создаёт обёртку QuantileEncoder на основе конфигурации."""
    cfg_dict = OmegaConf.to_container(factory_cfg, resolve=True)
    cfg_dict.pop("_target_", None)
    return QuantileEncoderWrapper(**cfg_dict)


def _build_encoder_transformers(cfg: DictConfig, encoder_cfg: DictConfig) -> list[tuple[str, Any, list[str]]]:
    """Создаёт список transformers для кодирования категориальных признаков."""
    if "encoders" in encoder_cfg:
        transformers = []

        for name, spec in encoder_cfg.encoders.items():
            column_key = CAT_COLUMN_KEYS.get(name)
            if column_key is not None:
                columns = _resolve_columns(cfg, column_key)
            else:
                columns = list(OmegaConf.to_container(spec.columns, resolve=True))

            target = spec.factory.get("_target_", "")
            if "OrdinalEncoder" in target:
                encoder = _instantiate_ordinal_encoder(spec.factory, columns)
            elif "QuantileEncoder" in target:
                encoder = _instantiate_quantile_encoder(spec.factory)
            else:
                encoder = hydra.utils.instantiate(spec.factory)

            transformers.append((name, encoder, columns))

        return transformers

    return []


def preprocessor(
    cfg: DictConfig,
    is_scale: bool = True,
    is_cat: bool = True,
    scaler: BaseEstimator | None = None,
    encoder_cfg: DictConfig | None = None,
) -> Pipeline | FeatureEngineer:
    """
    Создает пайплайн предобработки данных House Prices.

    Args:
        cfg: конфиг с секцией preprocessing
        is_scale: флаг, масштабировать ли числовые признаки
        is_cat: флаг, кодировать ли категориальные признаки
        scaler: sklearn-скейлер для числовых признаков
        encoder_cfg: конфиг энкодера
    """
    feature_engineer = FeatureEngineer()
    transformers = []

    if not is_cat and not is_scale:
        return feature_engineer

    if is_cat:
        transformers.extend(_build_encoder_transformers(cfg, encoder_cfg))

    if is_scale:
        numeric_cols = list(OmegaConf.to_container(cfg.preprocessing.num_cols, resolve=True))
        transformers.append(("num", scaler, numeric_cols))

    if not transformers:
        return Pipeline([("feature_engineering", feature_engineer)])

    cols_trans = ColumnTransformer(transformers, remainder="drop")

    return Pipeline(
        [("feature_engineering", feature_engineer), ("cols_transformer", cols_trans)]
    )


def build_preprocessor(
    cfg: DictConfig,
    model_cfg: DictConfig,
    is_scale: bool,
    is_cat: bool,
) -> Pipeline | FeatureEngineer:
    """Создаёт preprocessor с параметрами из конфига модели или глобального preprocessing."""
    scaler = None
    encoder_cfg = None

    if is_scale:
        scaler_cfg = model_cfg.get("scaler") or cfg.preprocessing.get("scaler")
        scaler = hydra.utils.instantiate(scaler_cfg)

    if is_cat:
        encoder_cfg = model_cfg.get("encoder") or cfg.preprocessing.get("encoder")

    return preprocessor(
        cfg=cfg,
        is_scale=is_scale,
        is_cat=is_cat,
        scaler=scaler,
        encoder_cfg=encoder_cfg,
    )


def pipeline_fit_params(model: Any, cat_features: list[str] | None = None) -> dict[str, Any]:
    """Возвращает fit-параметры только для моделей, поддерживающих cat_features."""
    if not cat_features:
        return {}

    fit_signature = inspect.signature(getattr(model, "fit", None))
    if "cat_features" in fit_signature.parameters:
        return {"model__cat_features": list(cat_features)}

    return {}
