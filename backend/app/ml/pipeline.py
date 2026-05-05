"""Build sklearn Pipeline cho dataset OULAD đã preprocess.

Schema của các cột (numeric / ordinal / nominal) đọc từ
`data/processed/feature_metadata.json` (sinh bởi notebook 01).

Cùng pipeline này được dùng trong:
- `notebooks/02_rf_pruning_ablation.ipynb` (thí nghiệm best params)
- `app.ml.train.run()` (huấn luyện model production)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from app.core.config import settings


@lru_cache(maxsize=1)
def load_feature_metadata() -> dict[str, Any]:
    path = settings.DATA_DIR / "processed" / "feature_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy notebooks/01_eda_preprocessing.ipynb "
            "để sinh feature_metadata.json."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer chuẩn cho RF: impute + ordinal/onehot encode (không scale)."""
    meta = load_feature_metadata()
    numeric_cols = meta["numeric_cols"]
    categorical_ordinal = meta["categorical_ordinal"]
    categorical_nominal = meta["categorical_nominal"]
    ordinal_categories = meta["ordinal_categories"]

    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("ord", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OrdinalEncoder(
                categories=[ordinal_categories[c] + ["Unknown"] for c in categorical_ordinal],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]), categorical_ordinal),
        ("nom", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_nominal),
    ])


def build_pipeline(rf_kwargs: dict[str, Any]) -> Pipeline:
    """Wrap preprocessor + RandomForestClassifier vào một Pipeline duy nhất."""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_kwargs)
    return Pipeline([("prep", build_preprocessor()), ("rf", rf)])


def load_best_params() -> dict[str, Any]:
    """Đọc best params từ notebook 02 (data/processed/best_rf_params.json).

    Nếu chưa có, trả về default (n_estimators=200, không pruning).
    """
    path = settings.DATA_DIR / "processed" / "best_rf_params.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        # max_depth có thể là string "None" hoặc int do JSON serialize
        if params.get("max_depth") in (None, "None", "null"):
            params["max_depth"] = None
        return params
    return {
        "n_estimators": 200,
        "max_depth": None,
        "ccp_alpha": 0.0,
        "min_impurity_decrease": 0.0,
        "random_state": 42,
        "n_jobs": -1,
    }
