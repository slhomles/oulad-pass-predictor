"""Smoke test cho `app.ml.train.run()`.

Skip nếu data đã preprocess (X.parquet, y.parquet) chưa có — vì pipeline mới
phụ thuộc vào output của notebook 01.

Resolve data path: `settings.DATA_DIR` mặc định là `./data` (relative). Khi
pytest chạy từ `backend/`, đường dẫn đó không trỏ tới data thật ở repo root.
Test này tự dò 2 ứng viên (`./data` và `../data`) và override `settings` cho
phiên test.
"""
from pathlib import Path

import joblib
import pytest

from app.core.config import settings
from app.ml import train
from app.ml.pipeline import load_feature_metadata


def _resolve_repo_data_dir() -> Path | None:
    for candidate in (Path("./data"), Path("../data")):
        if (candidate / "processed" / "X.parquet").exists():
            return candidate.resolve()
    return None


REPO_DATA = _resolve_repo_data_dir()


@pytest.mark.skipif(
    REPO_DATA is None,
    reason="data/processed/X.parquet|y.parquet chưa tồn tại (chạy notebook 01)",
)
def test_train_run_writes_model_and_meets_baseline(tmp_path):
    settings.DATA_DIR = REPO_DATA
    settings.MODEL_DIR = tmp_path / "models"
    load_feature_metadata.cache_clear()  # @lru_cache giữ feature_metadata cũ


    result = train.run()

    assert "accuracy" in result
    assert result["accuracy"] > 0.6, f"accuracy thấp bất thường: {result['accuracy']}"
    for key in ("f1", "precision", "recall", "timestamp", "params"):
        assert key in result

    latest = settings.MODEL_DIR / "rf_latest.pkl"
    assert latest.exists(), f"{latest} không được sinh ra"

    stamped = settings.MODEL_DIR / f"rf_{result['timestamp']}.pkl"
    assert stamped.exists(), f"{stamped} không được sinh ra"

    pipe = joblib.load(latest)
    assert hasattr(pipe, "predict_proba"), "model load lại phải có predict_proba"
