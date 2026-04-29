"""Load & preprocess OULAD dataset.

Fast path: đọc trực tiếp `data/processed/X.parquet` + `y.parquet` đã được sinh
ra bởi `notebooks/01_eda_preprocessing.ipynb`.

Fallback `load_raw()` (đọc 7 file CSV gốc và build features từ đầu) được hoãn
đến vòng làm việc sau — hiện tại raise NotImplementedError với thông điệp rõ.
"""
from pathlib import Path

import pandas as pd

from app.core.config import settings


def load_raw() -> dict[str, pd.DataFrame]:
    """Đọc 7 bảng OULAD gốc từ `settings.DATA_DIR / 'raw'`.

    TODO: port logic từ `notebooks/01_eda_preprocessing.ipynb` Section 2-7
    để cho phép `build_features()` chạy lại từ raw CSVs khi parquet mất.
    """
    raise NotImplementedError(
        "load_raw() chưa được implement. Hiện tại build_features() đọc trực "
        "tiếp data/processed/X.parquet (đã sinh từ notebook 01). Khi cần "
        "rebuild từ raw CSVs, implement function này theo Section 2-7 của "
        "notebooks/01_eda_preprocessing.ipynb."
    )


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """Trả về (X, y) sẵn sàng để fit Pipeline.

    Fast path: đọc parquet đã preprocess từ notebook 01.
    """
    processed_dir: Path = settings.DATA_DIR / "processed"
    x_path = processed_dir / "X.parquet"
    y_path = processed_dir / "y.parquet"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {x_path} hoặc {y_path}. "
            "Chạy notebooks/01_eda_preprocessing.ipynb trước để sinh ra "
            "X.parquet và y.parquet."
        )

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)["target"]
    return X, y
