"""Load model mới nhất và dự đoán pass/fail cho một mẫu học sinh."""
from functools import lru_cache

import joblib

from app.core.config import settings


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(settings.MODEL_DIR / "rf_latest.pkl")


def run(features: dict) -> dict:
    # TODO: chuyển features dict -> DataFrame đúng schema, predict + predict_proba
    raise NotImplementedError
