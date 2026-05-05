"""Train Random Forest model từ dữ liệu đã preprocess và lưu vào MODEL_DIR.

Sử dụng `app.ml.pipeline.build_pipeline` (ColumnTransformer + RF) và best params
từ `data/processed/best_rf_params.json` (do notebook 02 sinh ra). Nếu file chưa
tồn tại, dùng default trong `pipeline.load_best_params`.
"""
from datetime import datetime

import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.ml.pipeline import build_pipeline, load_best_params
from app.ml.preprocessing import build_features


def run() -> dict:
    X, y = build_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_kwargs = {k: v for k, v in load_best_params().items()
                 if k not in ("random_state", "n_jobs")}
    pipe = build_pipeline(rf_kwargs)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(pipe.score(X_test, y_test)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(pipe, settings.MODEL_DIR / f"rf_{stamp}.pkl")
    joblib.dump(pipe, settings.MODEL_DIR / "rf_latest.pkl")

    # NOTE: predict._load_model() có @lru_cache → process đang chạy sẽ vẫn dùng
    # model cũ sau khi retrain. Endpoint gọi train.run() phải tự gọi
    # predict._load_model.cache_clear() để invalidate cache.
    # Xem CLAUDE.md > "Known gotcha: stale model after retrain".

    return {**metrics, "timestamp": stamp, "params": rf_kwargs}
