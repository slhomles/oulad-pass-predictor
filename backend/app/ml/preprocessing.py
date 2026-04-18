"""Load & preprocess OULAD dataset.

Đọc các file CSV trong DATA_DIR/raw (studentInfo, studentVle, courses, ...),
merge & feature engineering, trả về X, y sẵn sàng để train.
"""
import pandas as pd

from app.core.config import settings


def load_raw() -> dict[str, pd.DataFrame]:
    # TODO: đọc các file OULAD (.csv) trong settings.DATA_DIR / "raw"
    raise NotImplementedError


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    # TODO: merge bảng, feature engineering, target = final_result in {Pass, Distinction}
    raise NotImplementedError
