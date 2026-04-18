"""Train Random Forest model từ dữ liệu đã preprocess và lưu vào MODEL_DIR."""
from datetime import datetime

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.ml.preprocessing import build_features


def run() -> dict:
    X, y = build_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, settings.MODEL_DIR / f"rf_{stamp}.pkl")
    joblib.dump(model, settings.MODEL_DIR / "rf_latest.pkl")

    return {"accuracy": score, "timestamp": stamp}
