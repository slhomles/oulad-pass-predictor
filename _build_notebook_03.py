"""Builder cho notebooks/03_rf_svm_comparison.ipynb.

Chạy: python _build_notebook_03.py
Sẽ sinh ra notebooks/03_rf_svm_comparison.ipynb (ghi đè nếu đã tồn tại).
Có thể xoá file builder này sau khi notebook đã được tạo thành công.
"""
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells: list[nbf.NotebookNode] = []

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
cells.append(md(r"""
# 03 — So sánh Random Forest vs SVM trên OULAD

**Mục tiêu:** huấn luyện, tinh chỉnh và so sánh hai mô hình **Random Forest** và **SVM** để chọn mô hình dự đoán pass/fail tốt nhất cho dataset OULAD.

Notebook này được tổ chức theo đúng 6 phần của báo cáo:

1. Thiết lập huấn luyện (Setup)
2. Tiền xử lý & phân nhánh pipeline cho 2 thuật toán
3. Đánh giá mô hình cơ sở (Baseline)
4. Tối ưu siêu tham số (Hyperparameter Tuning)
5. So sánh toàn diện & lựa chọn mô hình
6. Phân tích lỗi sâu (Deep Error Analysis)

Cộng thêm Section 7 lưu artifacts (bảng kết quả, params, biểu đồ) để chèn vào báo cáo.

**Dữ liệu đầu vào:** `data/processed/X.parquet` + `y.parquet` + `feature_metadata.json` (đã được sinh ra bởi `01_eda_preprocessing.ipynb`).

**Điểm quan trọng:** notebook đọc **dữ liệu thô chưa encode/scale** rồi build pipeline tự encode + scale bên trong CV fold → tránh data leakage.
"""))

# =============================================================================
# Section 1 — Setup
# =============================================================================
cells.append(md(r"""
## Section 1 — Thiết lập huấn luyện (Setup)

- **Công cụ:** `scikit-learn` (RandomForestClassifier, SVC, LinearSVC, StandardScaler, ColumnTransformer, CV utilities, metrics).
- **Chiến lược đánh giá:** Hold-out 80/20 (stratify theo nhãn) + **Stratified 5-Fold CV** trên tập train.
- **Độ đo:** **F1-score** quyết định, **ROC-AUC** bổ trợ, kèm Accuracy, Precision, Recall và đặc biệt là **fit time / inference time** (điểm khác biệt rất lớn giữa RF và SVM).
"""))

cells.append(code(r"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

RNG = 42
np.random.seed(RNG)

# Flags điều khiển khối lượng tính toán
RUN_HEAVY_TUNING = True   # False → skip Section 4 (dùng default/best params có sẵn)
SVM_TUNE_SUBSAMPLE = 10000   # subsample size để tune SVM (None = dùng full train set)
RF_SEARCH_N_ITER = 20
SVM_SEARCH_N_ITER = 20
SVM_TUNE_CV = 3   # CV folds cho tuning SVM (cv=5 quá đắt với 32k mẫu)

print(f"RUN_HEAVY_TUNING   = {RUN_HEAVY_TUNING}")
print(f"SVM_TUNE_SUBSAMPLE = {SVM_TUNE_SUBSAMPLE}")
"""))

cells.append(code(r"""
# Đường dẫn — notebook chạy từ thư mục notebooks/
NB_DIR = Path.cwd()
ROOT = NB_DIR.parent if NB_DIR.name == "notebooks" else NB_DIR
DATA = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print(f"ROOT = {ROOT}")
print(f"DATA = {DATA}")
print(f"FIG  = {FIG}")
"""))

cells.append(code(r"""
# Load dữ liệu đã preprocess (raw features chưa encode)
X = pd.read_parquet(DATA / "X.parquet")
y = pd.read_parquet(DATA / "y.parquet")["target"]

with open(DATA / "feature_metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Pass rate (y=1): {y.mean():.4f}")
print(f"Prediction day cutoff: {META.get('prediction_day', 'not set')}")
print(f"Excluded features: {META.get('excluded_features', [])}")
print(f"Time window rule: {META.get('time_window_rule', 'not set')}")
print()
print(f"Numeric cols   ({len(META['numeric_cols'])}): {META['numeric_cols']}")
print(f"Ordinal cols   ({len(META['categorical_ordinal'])}): {META['categorical_ordinal']}")
print(f"Nominal cols   ({len(META['categorical_nominal'])}): {META['categorical_nominal']}")
"""))

cells.append(code(r"""
# Hold-out split 80/20 (stratify theo nhãn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG
)

# Cross-validation chính cho đánh giá
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

print(f"Train set: {X_train.shape}  | pass rate = {y_train.mean():.4f}")
print(f"Test set:  {X_test.shape}   | pass rate = {y_test.mean():.4f}")
print(f"CV: StratifiedKFold(n_splits=5, shuffle=True, random_state={RNG})")
"""))

cells.append(code(r"""
# Trực quan: phân bố target + tỉ lệ train/test
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Phân bố nhãn trên toàn bộ dataset
counts = y.value_counts().sort_index()
axes[0].bar(["Fail/Withdrawn (0)", "Pass/Distinction (1)"],
            counts.values, color=["#d9534f", "#5cb85c"])
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v}\n({v / len(y) * 100:.1f}%)", ha="center", va="bottom")
axes[0].set_title(f"Phân bố nhãn — tổng {len(y)} mẫu")
axes[0].set_ylabel("Số mẫu")
axes[0].set_ylim(0, counts.max() * 1.15)

# (b) Train/test split (stacked bar)
split_data = pd.DataFrame({
    "Train (80%)": [(y_train == 0).sum(), (y_train == 1).sum()],
    "Test (20%)":  [(y_test == 0).sum(),  (y_test == 1).sum()],
}, index=["Fail (0)", "Pass (1)"])
split_data.T.plot(kind="bar", stacked=True, ax=axes[1],
                  color=["#d9534f", "#5cb85c"], width=0.6)
axes[1].set_title("Train/Test split (stratified)")
axes[1].set_ylabel("Số mẫu")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title="")

plt.tight_layout()
plt.savefig(FIG / "03_target_distribution.png")
plt.show()
"""))

# =============================================================================
# Section 2 — Pipeline
# =============================================================================
cells.append(md(r"""
## Section 2 — Tiền xử lý & Pipeline phân nhánh cho RF / SVM

### Vì sao phải phân nhánh pipeline?

| Yếu tố | Random Forest | SVM |
|---|---|---|
| Cơ chế phân loại | Cây quyết định + bỏ phiếu (ngưỡng axis-aligned) | Tìm siêu phẳng tối ưu trong không gian feature |
| Nhạy cảm với scale | **KHÔNG** — split chỉ phụ thuộc thứ tự | **CÓ** — tính khoảng cách, cột scale lớn sẽ át cột khác |
| Yêu cầu bắt buộc | Impute + Encode | Impute + Encode + **StandardScaler** |
| Curse of dimensionality | Ít nhạy hơn (subsample features per split) | Rất nhạy → cần giảm chiều / feature selection |

Notebook tách thành 2 factory function `make_rf_pipeline` và `make_svm_pipeline`. Cả 2 đều dùng chung 1 `preprocessor` (impute + encode), nhưng SVM chèn thêm bước `StandardScaler` giữa preprocessor và classifier để scaler chỉ fit trên train fold của mỗi vòng CV (không có data leakage).

Ngoài ra, cả 2 model đều dùng `class_weight='balanced'` để so sánh công bằng (mặc dù pass rate 47.2% gần như cân bằng — đây là quy ước báo cáo).
"""))

cells.append(code(r"""
def make_preprocessor() -> ColumnTransformer:
    '''Tạo ColumnTransformer dùng chung cho cả RF và SVM.

    Mỗi nhóm cột có chiến lược riêng:
      - numeric: SimpleImputer(median) — RF không cần scale, SVM sẽ scale ở bước sau
      - ordinal: SimpleImputer('Unknown') + OrdinalEncoder theo thứ tự tự nhiên
      - nominal: SimpleImputer('Unknown') + OneHotEncoder
    '''
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), META["numeric_cols"]),
        ("ord", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OrdinalEncoder(
                categories=[META["ordinal_categories"][c] + ["Unknown"]
                            for c in META["categorical_ordinal"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]), META["categorical_ordinal"]),
        ("nom", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), META["categorical_nominal"]),
    ])


def make_rf_pipeline(rf_kwargs: dict | None = None) -> Pipeline:
    '''Pipeline cho Random Forest — chỉ preprocessor + classifier (KHÔNG scale).'''
    rf_kwargs = rf_kwargs or {}
    defaults = dict(random_state=RNG, n_jobs=-1, class_weight="balanced")
    rf = RandomForestClassifier(**{**defaults, **rf_kwargs})
    return Pipeline([("prep", make_preprocessor()), ("clf", rf)])


def make_svm_pipeline(svm_kwargs: dict | None = None, linear: bool = False) -> Pipeline:
    '''Pipeline cho SVM — preprocessor + **StandardScaler** + classifier.

    Args:
        svm_kwargs: tham số cho SVC/LinearSVC.
        linear: True → dùng LinearSVC (nhanh hơn, nhưng không có predict_proba).
                False → dùng SVC(probability=True) để hỗ trợ ROC-AUC.
    '''
    svm_kwargs = svm_kwargs or {}
    if linear:
        defaults = dict(random_state=RNG, class_weight="balanced", max_iter=5000)
        clf = LinearSVC(**{**defaults, **svm_kwargs})
    else:
        defaults = dict(random_state=RNG, class_weight="balanced", probability=True)
        clf = SVC(**{**defaults, **svm_kwargs})
    return Pipeline([
        ("prep", make_preprocessor()),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
"""))

cells.append(code(r"""
# Smoke test cả 2 pipeline trên 500 dòng đầu để verify chạy được
_sample_X = X_train.head(500)
_sample_y = y_train.head(500)

_rf = make_rf_pipeline({"n_estimators": 10, "max_depth": 5})
_rf.fit(_sample_X, _sample_y)
print(f"RF pipeline OK — output shape sau preprocessor: "
      f"{_rf.named_steps['prep'].transform(_sample_X.head(5)).shape}")

_svm = make_svm_pipeline({"kernel": "rbf", "C": 1.0})
_svm.fit(_sample_X, _sample_y)
print(f"SVM pipeline OK — classes: {_svm.classes_}")

del _sample_X, _sample_y, _rf, _svm
"""))

cells.append(code(r"""
# Trực quan: vì sao SVM cần StandardScaler — so sánh phân bố numeric cols trước & sau scale
_prep_only = make_preprocessor().fit(X_train.head(5000))
_X_enc = _prep_only.transform(X_train.head(5000))
print(f"Encoded feature dimension: {_X_enc.shape[1]} "
      f"({len(META['numeric_cols'])} numeric raw columns)")
# Use the numeric column count from metadata and ColumnTransformer order
_X_num_raw = _X_enc[:, :len(META["numeric_cols"])]
_X_num_std = StandardScaler().fit_transform(_X_num_raw)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) before scaling: numeric columns can have very different scales
axes[0].boxplot([_X_num_raw[:, i] for i in range(len(META["numeric_cols"]))],
                labels=META["numeric_cols"], showfliers=False)
axes[0].set_xticklabels(META["numeric_cols"], rotation=75, fontsize=7)
axes[0].set_yscale("symlog")
axes[0].set_title("TRƯỚC scaling — scale rất khác nhau (RF không quan tâm)")
axes[0].set_ylabel("Giá trị (symlog)")
axes[0].grid(axis="y", alpha=0.3)

# (b) sau scale: cùng đơn vị (mean ~ 0, std ~ 1) — SVM cần điều này
axes[1].boxplot([_X_num_std[:, i] for i in range(len(META["numeric_cols"]))],
                labels=META["numeric_cols"], showfliers=False)
axes[1].set_xticklabels(META["numeric_cols"], rotation=75, fontsize=7)
axes[1].set_title("SAU StandardScaler — cùng scale (SVM cần)")
axes[1].set_ylabel("Z-score")
axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "03_scaling_before_after.png")
plt.show()
del _prep_only, _X_enc, _X_num_raw, _X_num_std
"""))

cells.append(md(r"""
> **Feature dimension note:** notebook 03 no longer hard-codes the encoded feature count. For the day-90 cutoff dataset, `last_active_day` is removed from metadata; raw feature counts and encoded dimensions are printed from `feature_metadata.json` and the fitted pipeline when the notebook is rerun.
"""))

# =============================================================================
# Section 3 — Baseline
# =============================================================================
cells.append(md(r"""
## Section 3 — Đánh giá Mô hình Cơ sở (Baseline)

Huấn luyện 2 mô hình với tham số mặc định để có **điểm xuất phát**:

- **Random Forest baseline:** `n_estimators=100`, `max_depth=None` (sklearn default) + `class_weight='balanced'`.
- **SVM baseline:** `kernel='rbf'`, `C=1.0`, `gamma='scale'`, `probability=True`, `class_weight='balanced'`.

Mỗi cấu hình chạy 5-Fold CV trên tập train, đo F1, ROC-AUC, Accuracy, Precision, Recall, fit_time, score_time, và gap giữa train score & CV score để đánh giá overfitting.
"""))

cells.append(code(r"""
SCORING = ["accuracy", "f1", "roc_auc", "precision", "recall"]


def eval_cv(pipe: Pipeline, name: str, cv=CV, X_=None, y_=None) -> dict:
    '''Chạy cross_validate và trả về dict metric phục vụ so sánh.'''
    X_ = X_train if X_ is None else X_
    y_ = y_train if y_ is None else y_
    t0 = time.time()
    cv_res = cross_validate(
        pipe, X_, y_,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    wall = time.time() - t0
    return {
        "name":            name,
        "f1_mean":         cv_res["test_f1"].mean(),
        "f1_std":          cv_res["test_f1"].std(),
        "auc_mean":        cv_res["test_roc_auc"].mean(),
        "auc_std":         cv_res["test_roc_auc"].std(),
        "acc_mean":        cv_res["test_accuracy"].mean(),
        "acc_std":         cv_res["test_accuracy"].std(),
        "precision_mean":  cv_res["test_precision"].mean(),
        "recall_mean":     cv_res["test_recall"].mean(),
        "train_f1_mean":   cv_res["train_f1"].mean(),
        "train_acc_mean":  cv_res["train_accuracy"].mean(),
        "overfit_gap_acc": cv_res["train_accuracy"].mean() - cv_res["test_accuracy"].mean(),
        "fit_time_mean":   cv_res["fit_time"].mean(),
        "score_time_mean": cv_res["score_time"].mean(),
        "wall_time_total": wall,
        # raw per-fold arrays (cho boxplot, variance analysis)
        "f1_scores":       cv_res["test_f1"].tolist(),
        "auc_scores":      cv_res["test_roc_auc"].tolist(),
        "acc_scores":      cv_res["test_accuracy"].tolist(),
        "train_f1_scores": cv_res["train_f1"].tolist(),
    }
"""))

cells.append(code(r"""
# Baseline Random Forest
print("=" * 60)
print("Baseline Random Forest — 5-Fold CV")
print("=" * 60)
baseline_rf_pipe = make_rf_pipeline()   # default sklearn + class_weight='balanced'
res_baseline_rf = eval_cv(baseline_rf_pipe, "baseline_rf")
print(f"  F1     = {res_baseline_rf['f1_mean']:.4f} ± {res_baseline_rf['f1_std']:.4f}")
print(f"  AUC    = {res_baseline_rf['auc_mean']:.4f} ± {res_baseline_rf['auc_std']:.4f}")
print(f"  Acc    = {res_baseline_rf['acc_mean']:.4f} ± {res_baseline_rf['acc_std']:.4f}")
print(f"  Fit    = {res_baseline_rf['fit_time_mean']:.2f}s per fold")
print(f"  Train acc - CV acc = {res_baseline_rf['overfit_gap_acc']:+.4f}  (gap → overfit nếu lớn)")
"""))

cells.append(code(r"""
# Baseline SVM — CẢNH BÁO: bước này chậm (~vài phút trên 32k mẫu × 5 folds)
print("=" * 60)
print("Baseline SVM (RBF, C=1.0) — 5-Fold CV")
print("=" * 60)
baseline_svm_pipe = make_svm_pipeline({"kernel": "rbf", "C": 1.0, "gamma": "scale"})
res_baseline_svm = eval_cv(baseline_svm_pipe, "baseline_svm")
print(f"  F1     = {res_baseline_svm['f1_mean']:.4f} ± {res_baseline_svm['f1_std']:.4f}")
print(f"  AUC    = {res_baseline_svm['auc_mean']:.4f} ± {res_baseline_svm['auc_std']:.4f}")
print(f"  Acc    = {res_baseline_svm['acc_mean']:.4f} ± {res_baseline_svm['acc_std']:.4f}")
print(f"  Fit    = {res_baseline_svm['fit_time_mean']:.2f}s per fold")
print(f"  Train acc - CV acc = {res_baseline_svm['overfit_gap_acc']:+.4f}")
"""))

cells.append(code(r"""
# Tổng hợp bảng baseline
baseline_df = pd.DataFrame([res_baseline_rf, res_baseline_svm]).set_index("name")
display(baseline_df.round(4))
"""))

cells.append(code(r"""
# Bar chart so sánh metric + fit time
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

METRIC_KEYS = ["f1_mean", "auc_mean", "acc_mean"]
ERR_KEYS    = ["f1_std",  "auc_std",  "acc_std"]
LABELS      = ["F1", "ROC-AUC", "Accuracy"]
COLORS      = {"baseline_rf": "tab:blue", "baseline_svm": "tab:orange"}

x = np.arange(len(METRIC_KEYS))
width = 0.36
for i, (name, row) in enumerate(baseline_df.iterrows()):
    means = [row[m] for m in METRIC_KEYS]
    stds  = [row[e] for e in ERR_KEYS]
    axes[0].bar(x + (i - 0.5) * width, means, width, yerr=stds,
                label=name, color=COLORS[name], capsize=4)
axes[0].set_xticks(x)
axes[0].set_xticklabels(LABELS)
axes[0].set_ylim(0.4, 1.0)
axes[0].set_ylabel("Score")
axes[0].set_title("Baseline: 5-Fold CV scores")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

ft = baseline_df["fit_time_mean"]
axes[1].bar(ft.index, ft.values, color=[COLORS[n] for n in ft.index])
axes[1].set_yscale("log")
axes[1].set_ylabel("Fit time per fold (s, log scale)")
axes[1].set_title("Baseline: chi phí huấn luyện")
for i, v in enumerate(ft.values):
    axes[1].text(i, v, f"{v:.1f}s", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(FIG / "03_baseline_compare.png")
plt.show()
"""))

cells.append(code(r"""
# Per-fold F1 boxplot — kiểm tra ổn định (variance giữa các fold)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

fold_data = pd.DataFrame({
    "baseline_rf":  res_baseline_rf["f1_scores"],
    "baseline_svm": res_baseline_svm["f1_scores"],
})
axes[0].boxplot([fold_data["baseline_rf"], fold_data["baseline_svm"]],
                labels=["RF", "SVM"], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
for i, name in enumerate(["baseline_rf", "baseline_svm"]):
    scores = fold_data[name].values
    axes[0].scatter([i + 1] * len(scores), scores, color="red", zorder=10, s=30)
axes[0].set_ylabel("F1 score per fold")
axes[0].set_title("Phân bố F1 qua 5 folds (chấm đỏ = từng fold)")
axes[0].grid(axis="y", alpha=0.3)

# Train vs CV F1 (gap visualization)
x = np.arange(2)
width = 0.35
axes[1].bar(x - width / 2,
            [res_baseline_rf["train_f1_mean"], res_baseline_svm["train_f1_mean"]],
            width, label="Train F1", color="steelblue")
axes[1].bar(x + width / 2,
            [res_baseline_rf["f1_mean"], res_baseline_svm["f1_mean"]],
            width, label="CV F1", color="lightcoral")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["RF", "SVM"])
axes[1].set_ylim(0.4, 1.0)
axes[1].set_title("Train F1 vs CV F1 (gap → overfit)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)
for i, (tr, cv) in enumerate([
    (res_baseline_rf["train_f1_mean"], res_baseline_rf["f1_mean"]),
    (res_baseline_svm["train_f1_mean"], res_baseline_svm["f1_mean"]),
]):
    gap = tr - cv
    axes[1].text(i, max(tr, cv) + 0.02, f"gap={gap:+.3f}",
                 ha="center", fontsize=9, color="darkred" if gap > 0.05 else "darkgreen")

plt.tight_layout()
plt.savefig(FIG / "03_baseline_per_fold.png")
plt.show()
"""))

cells.append(md(r"""
**Nhận xét overfit:** Random Forest mặc định (`max_depth=None`) thường đạt train accuracy ≈ 1.0 → gap train-CV rất lớn → biểu hiện overfit (sẽ xử lý bằng pruning ở phần tuning). SVM RBF với C=1.0 thường có gap nhỏ hơn nhờ regularization tích hợp.
"""))

# =============================================================================
# Section 4 — Tuning
# =============================================================================
cells.append(md(r"""
## Section 4 — Tối ưu siêu tham số (Hyperparameter Tuning)

### 4.1 — RF tuning
Không gian search: `n_estimators × max_depth × min_samples_split × ccp_alpha`. Dùng `RandomizedSearchCV(n_iter=20, cv=5)` để cân bằng giữa độ phủ và thời gian.

### 4.2 — SVM tuning (chi phí tính toán)
SVM RBF có độ phức tạp **O(n²) – O(n³)** theo số mẫu → trên 32k mẫu × 5 folds × N candidates là không khả thi nếu dùng GridSearchCV. Giải pháp của notebook:

- Dùng `RandomizedSearchCV` thay vì GridSearch để giới hạn số candidate.
- Hạ `cv=3` thay vì `cv=5` trong vòng tuning (giảm 40% chi phí).
- Tune trên **subsample 10k mẫu stratified**, sau đó refit best params trên full train. Best params trên subsample thường cùng hạng với best params trên full set, nhưng chi phí tune giảm ~10 lần.
"""))

cells.append(code(r"""
# 4.1 — Tuning Random Forest
if RUN_HEAVY_TUNING:
    rf_grid = {
        "clf__n_estimators":       [100, 200, 300],
        "clf__max_depth":          [10, 15, 20, None],
        "clf__min_samples_split":  [2, 5, 10],
        "clf__ccp_alpha":          [0.0, 1e-4, 1e-3],
    }
    print(f"RF RandomizedSearchCV: n_iter={RF_SEARCH_N_ITER}, cv=5, scoring=f1")
    rf_search = RandomizedSearchCV(
        make_rf_pipeline(),
        rf_grid,
        n_iter=RF_SEARCH_N_ITER,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=RNG,
        refit=True,
        verbose=1,
    )
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    rf_search_time = time.time() - t0
    rf_best_params = {k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()}
    print(f"\n  Took {rf_search_time:.1f}s. Best F1 = {rf_search.best_score_:.4f}")
    print(f"  Best params: {rf_best_params}")
else:
    # Fallback: dùng best_rf_params.json đã có
    with open(DATA / "best_rf_params.json") as f:
        rf_best_params = {k: v for k, v in json.load(f).items()
                          if k not in ("random_state", "n_jobs")}
    rf_search = None
    print(f"Skipping RF tuning. Loaded existing best params: {rf_best_params}")
"""))

cells.append(code(r"""
# 4.2 — Tuning SVM (subsample + RandomizedSearchCV + cv=3)
if RUN_HEAVY_TUNING:
    if SVM_TUNE_SUBSAMPLE and SVM_TUNE_SUBSAMPLE < len(X_train):
        # Stratified subsample
        sub_idx = (
            pd.Series(y_train.index, index=y_train.values)
            .groupby(level=0, group_keys=False)
            .apply(lambda s: s.sample(
                n=int(SVM_TUNE_SUBSAMPLE * (y_train == s.name).mean()),
                random_state=RNG,
            ))
            .values
        )
        X_sub, y_sub = X_train.loc[sub_idx], y_train.loc[sub_idx]
        print(f"SVM tuning subsample: n={len(X_sub)} (pass rate = {y_sub.mean():.3f})")
    else:
        X_sub, y_sub = X_train, y_train
        print(f"SVM tuning on full train set: n={len(X_sub)}")

    svm_grid = {
        "clf__C":      [0.1, 1.0, 10.0, 100.0],
        "clf__kernel": ["linear", "rbf", "poly"],
        "clf__gamma":  ["scale", "auto", 0.01, 0.1],
    }
    print(f"SVM RandomizedSearchCV: n_iter={SVM_SEARCH_N_ITER}, cv={SVM_TUNE_CV}, scoring=f1")
    svm_search = RandomizedSearchCV(
        make_svm_pipeline(),
        svm_grid,
        n_iter=SVM_SEARCH_N_ITER,
        cv=SVM_TUNE_CV,
        scoring="f1",
        n_jobs=-1,
        random_state=RNG,
        refit=False,   # ta sẽ refit trên full train set sau
        verbose=1,
    )
    t0 = time.time()
    svm_search.fit(X_sub, y_sub)
    svm_search_time = time.time() - t0
    svm_best_params = {k.replace("clf__", ""): v for k, v in svm_search.best_params_.items()}
    print(f"\n  Took {svm_search_time:.1f}s. Best F1 (subsample) = {svm_search.best_score_:.4f}")
    print(f"  Best params: {svm_best_params}")
else:
    svm_best_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
    svm_search = None
    print(f"Skipping SVM tuning. Using defaults: {svm_best_params}")
"""))

cells.append(code(r"""
# Visualize top-5 candidates cho mỗi model
if RUN_HEAVY_TUNING and rf_search is not None and svm_search is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    rf_cv = pd.DataFrame(rf_search.cv_results_)
    rf_top5 = rf_cv.nlargest(5, "mean_test_score")[["params", "mean_test_score", "std_test_score"]]
    rf_labels = [str({k.replace("clf__", ""): v for k, v in p.items()}) for p in rf_top5["params"]]
    axes[0].barh(range(5), rf_top5["mean_test_score"], xerr=rf_top5["std_test_score"],
                 color="tab:blue", alpha=0.85, capsize=3)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(rf_labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean F1 (5-fold CV)")
    axes[0].set_title("RF — Top 5 candidates")
    axes[0].grid(axis="x", alpha=0.3)

    svm_cv = pd.DataFrame(svm_search.cv_results_)
    svm_top5 = svm_cv.nlargest(5, "mean_test_score")[["params", "mean_test_score", "std_test_score"]]
    svm_labels = [str({k.replace("clf__", ""): v for k, v in p.items()}) for p in svm_top5["params"]]
    axes[1].barh(range(5), svm_top5["mean_test_score"], xerr=svm_top5["std_test_score"],
                 color="tab:orange", alpha=0.85, capsize=3)
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(svm_labels, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"Mean F1 ({SVM_TUNE_CV}-fold CV, subsample n={SVM_TUNE_SUBSAMPLE})")
    axes[1].set_title("SVM — Top 5 candidates")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / "03_hyperparam_top5.png")
    plt.show()
else:
    print("Bỏ qua biểu đồ top-5 (RUN_HEAVY_TUNING=False).")
"""))

cells.append(code(r"""
# Lưu best SVM params (ghi đè placeholder cũ {C: 1.0})
with open(DATA / "best_svm_params.json", "w", encoding="utf-8") as f:
    json.dump(svm_best_params, f, indent=2)
print(f"Saved → {DATA / 'best_svm_params.json'}")
print(f"  {svm_best_params}")
"""))

# =============================================================================
# Section 5 — Champion vs Champion
# =============================================================================
cells.append(md(r"""
## Section 5 — So sánh toàn diện & lựa chọn mô hình (Champion vs Champion)

Đặt 2 mô hình tốt nhất (best params) đối đầu trên cùng cấu hình CV để chọn winner. Sau đó refit trên full train set và đánh giá trên test set 20% giữ kín.
"""))

cells.append(code(r"""
champion_rf_pipe = make_rf_pipeline(rf_best_params)
champion_svm_pipe = make_svm_pipeline(svm_best_params)
print("Champion pipelines built with best params.")
print(f"  RF  : {rf_best_params}")
print(f"  SVM : {svm_best_params}")
"""))

cells.append(md(r"""
### 5.1 — Learning Curves (bắt buộc cho cả 2 thuật toán)

Learning curve trả lời 3 câu hỏi:

1. **Mô hình có đang underfit / overfit không?**
   - Train score cao + CV score thấp → variance (overfit)
   - Train score và CV score đều thấp → bias (underfit)
2. **Thêm dữ liệu có giúp không?**
   - CV score còn đang tăng → có thể thêm data
   - CV score plateau → đã bão hoà, cần thay feature hoặc model
3. **Hai mô hình cần lượng dữ liệu khác nhau để đạt cùng performance không?**

**Lưu ý chi phí:** SVM RBF có độ phức tạp O(n²)-O(n³) → learning curve trên 32k mẫu × nhiều train_sizes × nhiều CV folds là cực đắt. Notebook dùng subsample 8000 cho SVM learning curve, full data cho RF.
"""))

cells.append(code(r"""
# Cấu hình learning curve
LC_TRAIN_SIZES = np.array([0.02, 0.05, 0.10, 0.30, 0.60, 1.00])
LC_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
LC_SCORING = "f1"
# Keep learning-curve folds consistent with the main shuffled stratified CV.
LC_SVM_SUBSAMPLE = 8000   # giảm chi phí compute cho SVM

print("=" * 60)
print("Learning curve: Random Forest (full train set)")
print("=" * 60)
t0 = time.time()
rf_lc_sizes, rf_lc_train, rf_lc_test = learning_curve(
    champion_rf_pipe, X_train, y_train,
    train_sizes=LC_TRAIN_SIZES,
    cv=LC_CV,
    scoring=LC_SCORING,
    n_jobs=-1,
    random_state=RNG,
    shuffle=True,
)
print(f"  Done in {time.time() - t0:.1f}s")
print(f"  Train sizes used: {rf_lc_sizes}")
"""))

cells.append(code(r"""
print("=" * 60)
print(f"Learning curve: SVM (subsample n={LC_SVM_SUBSAMPLE} để giảm chi phí)")
print("=" * 60)
# Stratified subsample
_svm_lc_X = X_train.sample(n=min(LC_SVM_SUBSAMPLE, len(X_train)), random_state=RNG)
_svm_lc_y = y_train.loc[_svm_lc_X.index]

t0 = time.time()
svm_lc_sizes, svm_lc_train, svm_lc_test = learning_curve(
    champion_svm_pipe, _svm_lc_X, _svm_lc_y,
    train_sizes=LC_TRAIN_SIZES,
    cv=LC_CV,
    scoring=LC_SCORING,
    n_jobs=-1,
    random_state=RNG,
    shuffle=True,
)
print(f"  Done in {time.time() - t0:.1f}s")
print(f"  Train sizes used: {svm_lc_sizes}")
"""))

cells.append(code(r"""
# Plot learning curves side-by-side
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, sizes, train_s, test_s, name, color_train, color_test in [
    (axes[0], rf_lc_sizes, rf_lc_train, rf_lc_test,
     "Random Forest", "tab:blue", "tab:orange"),
    (axes[1], svm_lc_sizes, svm_lc_train, svm_lc_test,
     f"SVM (subsample n={LC_SVM_SUBSAMPLE})", "tab:blue", "tab:orange"),
]:
    train_mean = train_s.mean(axis=1); train_std = train_s.std(axis=1)
    test_mean  = test_s.mean(axis=1);  test_std  = test_s.std(axis=1)
    ax.plot(sizes, train_mean, "o-", color=color_train, label="Train F1", linewidth=2)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=color_train)
    ax.plot(sizes, test_mean, "s-", color=color_test, label="CV F1", linewidth=2)
    ax.fill_between(sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color=color_test)

    # Annotate final gap
    final_gap = train_mean[-1] - test_mean[-1]
    ax.annotate(
        f"Gap @ max size = {final_gap:+.3f}",
        xy=(sizes[-1], (train_mean[-1] + test_mean[-1]) / 2),
        xytext=(0.55, 0.5), textcoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"),
        fontsize=9,
    )

    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Learning curve — {name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.02)

plt.tight_layout()
plt.savefig(FIG / "03_learning_curves.png")
plt.show()
"""))

cells.append(code(r"""
# Bảng tóm tắt learning curve để chèn báo cáo
lc_summary = pd.DataFrame({
    "train_size": rf_lc_sizes,
    "RF_train_f1":  rf_lc_train.mean(axis=1).round(4),
    "RF_cv_f1":     rf_lc_test.mean(axis=1).round(4),
    "RF_gap":       (rf_lc_train.mean(axis=1) - rf_lc_test.mean(axis=1)).round(4),
})
svm_summary = pd.DataFrame({
    "svm_train_size": svm_lc_sizes,
    "SVM_train_f1": svm_lc_train.mean(axis=1).round(4),
    "SVM_cv_f1":    svm_lc_test.mean(axis=1).round(4),
    "SVM_gap":      (svm_lc_train.mean(axis=1) - svm_lc_test.mean(axis=1)).round(4),
})
print("Learning curve summary — RF:")
display(lc_summary)
print("\nLearning curve summary — SVM:")
display(svm_summary)
"""))

cells.append(md(r"""
### 5.2 — Champion vs Champion (5-Fold CV trên full train set)
"""))

cells.append(code(r"""
print("Champion RF — 5-Fold CV trên full train set...")
res_champ_rf = eval_cv(champion_rf_pipe, "champion_rf")
print(f"  F1 = {res_champ_rf['f1_mean']:.4f} ± {res_champ_rf['f1_std']:.4f}, "
      f"AUC = {res_champ_rf['auc_mean']:.4f}, fit = {res_champ_rf['fit_time_mean']:.2f}s/fold")

print("\nChampion SVM — 5-Fold CV trên full train set...")
res_champ_svm = eval_cv(champion_svm_pipe, "champion_svm")
print(f"  F1 = {res_champ_svm['f1_mean']:.4f} ± {res_champ_svm['f1_std']:.4f}, "
      f"AUC = {res_champ_svm['auc_mean']:.4f}, fit = {res_champ_svm['fit_time_mean']:.2f}s/fold")

champion_df = pd.DataFrame([res_champ_rf, res_champ_svm]).set_index("name")
display(champion_df.round(4))
"""))

cells.append(code(r"""
# Biểu đồ so sánh champion
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(METRIC_KEYS))
width = 0.36
COLOR2 = {"champion_rf": "tab:blue", "champion_svm": "tab:orange"}
for i, (name, row) in enumerate(champion_df.iterrows()):
    means = [row[m] for m in METRIC_KEYS]
    stds  = [row[e] for e in ERR_KEYS]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
           label=name, color=COLOR2[name], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Score")
ax.set_title("Champion vs Champion — 5-Fold CV")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "03_champion_compare.png")
plt.show()
"""))

cells.append(code(r"""
# Quyết định winner
def _gain(a: float, b: float) -> str:
    return f"{(a - b) * 100:+.2f}pp"

winner = "champion_rf" if res_champ_rf["f1_mean"] >= res_champ_svm["f1_mean"] else "champion_svm"
print("=" * 60)
print("Decision rationale")
print("=" * 60)
print(f"F1 (mean):    RF = {res_champ_rf['f1_mean']:.4f}, "
      f"SVM = {res_champ_svm['f1_mean']:.4f}  "
      f"(RF - SVM = {_gain(res_champ_rf['f1_mean'], res_champ_svm['f1_mean'])})")
print(f"Stability:    RF std = {res_champ_rf['f1_std']:.4f}, "
      f"SVM std = {res_champ_svm['f1_std']:.4f}")
print(f"Cost:         RF fit = {res_champ_rf['fit_time_mean']:.1f}s/fold, "
      f"SVM fit = {res_champ_svm['fit_time_mean']:.1f}s/fold")
print(f"Inference:    RF score_time = {res_champ_rf['score_time_mean']:.2f}s, "
      f"SVM score_time = {res_champ_svm['score_time_mean']:.2f}s")
print(f"\n>>> Winner theo F1: {winner}")
"""))

cells.append(md(r"""
### 5.2 — Đánh giá trên Test set (20% holdout)

Refit cả 2 champion trên full train set rồi predict trên test set. Đây là con số "trung thực" để đưa vào báo cáo.
"""))

cells.append(code(r"""
print("Refit champion_rf trên X_train (full)...")
t0 = time.time(); champion_rf_pipe.fit(X_train, y_train); rf_test_fit_time = time.time() - t0
print(f"  fit time = {rf_test_fit_time:.1f}s")

print("Refit champion_svm trên X_train (full)...")
t0 = time.time(); champion_svm_pipe.fit(X_train, y_train); svm_test_fit_time = time.time() - t0
print(f"  fit time = {svm_test_fit_time:.1f}s")

# Predict
t0 = time.time(); rf_test_pred = champion_rf_pipe.predict(X_test); rf_pred_time = time.time() - t0
rf_test_proba = champion_rf_pipe.predict_proba(X_test)[:, 1]
t0 = time.time(); svm_test_pred = champion_svm_pipe.predict(X_test); svm_pred_time = time.time() - t0
svm_test_proba = champion_svm_pipe.predict_proba(X_test)[:, 1]
print(f"\nInference time on {len(X_test)} samples: RF = {rf_pred_time:.2f}s, SVM = {svm_pred_time:.2f}s")
"""))

cells.append(code(r"""
test_results = pd.DataFrame([
    {
        "name": "test_rf",
        "f1":        f1_score(y_test, rf_test_pred),
        "auc":       roc_auc_score(y_test, rf_test_proba),
        "accuracy":  accuracy_score(y_test, rf_test_pred),
        "precision": precision_score(y_test, rf_test_pred),
        "recall":    recall_score(y_test, rf_test_pred),
        "fit_time":  rf_test_fit_time,
        "infer_time": rf_pred_time,
    },
    {
        "name": "test_svm",
        "f1":        f1_score(y_test, svm_test_pred),
        "auc":       roc_auc_score(y_test, svm_test_proba),
        "accuracy":  accuracy_score(y_test, svm_test_pred),
        "precision": precision_score(y_test, svm_test_pred),
        "recall":    recall_score(y_test, svm_test_pred),
        "fit_time":  svm_test_fit_time,
        "infer_time": svm_pred_time,
    },
]).set_index("name")
print("=" * 60)
print("Test set (20% holdout)")
print("=" * 60)
display(test_results.round(4))
"""))

cells.append(code(r"""
print("--- Classification report: champion_rf ---")
print(classification_report(
    y_test, rf_test_pred,
    target_names=["Fail/Withdrawn (0)", "Pass/Distinction (1)"],
    digits=4,
))
print("\n--- Classification report: champion_svm ---")
print(classification_report(
    y_test, svm_test_pred,
    target_names=["Fail/Withdrawn (0)", "Pass/Distinction (1)"],
    digits=4,
))
"""))

# =============================================================================
# Section 6 — Error Analysis
# =============================================================================
cells.append(md(r"""
## Section 6 — Phân tích lỗi sâu (Deep Error Analysis)

So sánh hai mô hình mắc lỗi như thế nào: confusion matrix, các vùng disagreement, ROC curves chồng lên nhau, và minh họa decision boundary trên PCA 2D.
"""))

cells.append(code(r"""
# 6.1 — Confusion matrices side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_test_pred,
    display_labels=["Fail", "Pass"],
    cmap="Blues", ax=axes[0], colorbar=False,
)
axes[0].set_title(f"RF — F1 = {test_results.loc['test_rf', 'f1']:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, svm_test_pred,
    display_labels=["Fail", "Pass"],
    cmap="Oranges", ax=axes[1], colorbar=False,
)
axes[1].set_title(f"SVM — F1 = {test_results.loc['test_svm', 'f1']:.4f}")

plt.tight_layout()
plt.savefig(FIG / "03_confusion_matrices.png")
plt.show()
"""))

cells.append(code(r"""
# 6.2 — Disagreement analysis: 4 nhóm
both_correct     = (rf_test_pred == y_test.values) & (svm_test_pred == y_test.values)
both_wrong       = (rf_test_pred != y_test.values) & (svm_test_pred != y_test.values)
only_rf_correct  = (rf_test_pred == y_test.values) & (svm_test_pred != y_test.values)
only_svm_correct = (rf_test_pred != y_test.values) & (svm_test_pred == y_test.values)

summary = pd.DataFrame({
    "n_samples": [both_correct.sum(), both_wrong.sum(),
                  only_rf_correct.sum(), only_svm_correct.sum()],
}, index=["Both correct", "Both wrong",
          "Only RF correct", "Only SVM correct"])
summary["pct"] = (summary["n_samples"] / len(y_test) * 100).round(2)
print("=== Phân chia 4 nhóm theo agreement ===")
display(summary)

# Cross-tab dạng confusion matrix giữa 2 model
cross = pd.crosstab(
    pd.Series(rf_test_pred, name="RF prediction"),
    pd.Series(svm_test_pred, name="SVM prediction"),
)
print("\n=== RF prediction × SVM prediction (cross-tab trên test set) ===")
display(cross)
"""))

cells.append(code(r"""
# 6.3 — Feature mean theo từng nhóm (chỉ numeric cols)
X_test_aug = X_test.copy()
X_test_aug["_group"] = "other"
X_test_aug.loc[X_test.index[both_correct],     "_group"] = "Both correct"
X_test_aug.loc[X_test.index[both_wrong],       "_group"] = "Both wrong"
X_test_aug.loc[X_test.index[only_rf_correct],  "_group"] = "Only RF"
X_test_aug.loc[X_test.index[only_svm_correct], "_group"] = "Only SVM"

group_means = (
    X_test_aug.groupby("_group")[META["numeric_cols"]]
    .mean()
    .round(2)
    .T
)
print("Feature mean theo từng nhóm (sinh viên):")
display(group_means)
"""))

cells.append(code(r"""
# 6.4 — ROC curves overlay
fig, ax = plt.subplots(figsize=(7, 6))
for label, proba, color in [
    ("RF",  rf_test_proba,  "tab:blue"),
    ("SVM", svm_test_proba, "tab:orange"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves trên test set")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "03_roc_curves.png")
plt.show()
"""))

cells.append(code(r"""
# 6.5 — Phân bố xác suất dự đoán theo nhãn thật
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

bins = np.linspace(0, 1, 41)
for ax, proba, name, color_correct in [
    (axes[0], rf_test_proba,  "RF",  "tab:blue"),
    (axes[1], svm_test_proba, "SVM", "tab:orange"),
]:
    proba_pos = proba[y_test == 1]   # sinh viên thực sự pass
    proba_neg = proba[y_test == 0]   # sinh viên thực sự fail
    ax.hist(proba_neg, bins=bins, alpha=0.55,
            label=f"Fail thật (n={len(proba_neg)})", color="#d9534f")
    ax.hist(proba_pos, bins=bins, alpha=0.55,
            label=f"Pass thật (n={len(proba_pos)})", color="#5cb85c")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Ngưỡng 0.5")
    ax.set_xlabel("P(pass) — xác suất dự đoán")
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"{name} — phân bố P(pass) theo nhãn thật")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG / "03_prob_histogram.png")
plt.show()
"""))

cells.append(code(r"""
# 6.6 — Visualize decision boundary trên PCA 2D
# Fit preprocessor + scaler trên train rồi PCA → 2D
_prep = make_preprocessor()
X_train_enc = _prep.fit_transform(X_train)
X_test_enc  = _prep.transform(X_test)

_scaler = StandardScaler().fit(X_train_enc)
X_train_std = _scaler.transform(X_train_enc)
X_test_std  = _scaler.transform(X_test_enc)

pca = PCA(n_components=2, random_state=RNG)
X_train_2d = pca.fit_transform(X_train_std)
X_test_2d  = pca.transform(X_test_std)
print(f"PCA explained variance ratio (2 components): {pca.explained_variance_ratio_.round(3)}, "
      f"total = {pca.explained_variance_ratio_.sum():.3f}")
"""))

cells.append(code(r"""
# Fit 2 mô hình DEMO (không phải champion thật) trên không gian 2D để vẽ ranh giới
rf_2d = RandomForestClassifier(
    n_estimators=100, max_depth=15, class_weight="balanced",
    random_state=RNG, n_jobs=-1,
).fit(X_train_2d, y_train)
svm_2d = SVC(
    kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RNG,
).fit(X_train_2d, y_train)

# Mesh grid
x_min, x_max = X_train_2d[:, 0].min() - 0.5, X_train_2d[:, 0].max() + 0.5
y_min, y_max = X_train_2d[:, 1].min() - 0.5, X_train_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                     np.linspace(y_min, y_max, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
Z_rf  = rf_2d.predict(grid).reshape(xx.shape)
Z_svm = svm_2d.predict(grid).reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
rng_sample = np.random.default_rng(RNG)
sample_idx = rng_sample.choice(len(X_train_2d), size=2500, replace=False)

for ax, Z, title in [
    (axes[0], Z_rf,  "RF — axis-aligned splits"),
    (axes[1], Z_svm, "SVM (RBF) — smooth nonlinear boundary"),
]:
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu_r", levels=1)
    ax.scatter(
        X_train_2d[sample_idx, 0], X_train_2d[sample_idx, 1],
        c=y_train.iloc[sample_idx], s=4, alpha=0.45,
        cmap="RdBu_r", edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(FIG / "03_pca_decision_boundary.png")
plt.show()
"""))

cells.append(md(r"""
### Đối chiếu bản chất thuật toán & đề xuất cải tiến

- **Pattern lỗi:** RF chia không gian bằng các đường vuông góc với trục feature (axis-aligned splits) → khi 2 lớp giao nhau theo đường chéo, RF cần nhiều cây sâu mới xấp xỉ tốt. SVM RBF tạo ranh giới phi tuyến **mượt mà** nhờ kernel trick, nên có xu hướng tốt hơn ở các vùng "ranh giới mờ" (sinh viên ở giáp ranh pass/fail).
- **Trade-off chi phí:** SVM RBF có chi phí huấn luyện O(n²)–O(n³) → trên dataset lớn hơn nhiều (vd 100k+ mẫu) thì RF sẽ vượt trội về tính khả mở rộng.
- **Đề xuất cải tiến:**
  1. **Feature engineering động học:** thêm các đặc trưng theo tuần học (clicks_week1, clicks_week2, …) để mô hình bắt được pattern bỏ học sớm.
  2. **Sequence-aware features:** trend của clicks/scores qua thời gian thay vì chỉ aggregate tổng.
  3. **Ensemble Stacking:** RF + SVM + LogisticRegression(meta) — tận dụng đặc tính bổ sung của 2 model (xem section optional bên dưới).
"""))

cells.append(code(r"""
# 6.7 — Optional: Stacking ensemble demo (gộp 2 champion)
# Cảnh báo: bước này tốn thời gian (fit cả 2 base models + meta learner).
# Có thể skip bằng cách set RUN_STACKING = False.
RUN_STACKING = True

if RUN_STACKING:
    print("Building Stacking ensemble (RF + SVM) → LogisticRegression meta...")
    stacking = StackingClassifier(
        estimators=[
            ("rf",  make_rf_pipeline(rf_best_params)),
            ("svm", make_svm_pipeline(svm_best_params)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RNG),
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )
    t0 = time.time()
    stacking.fit(X_train, y_train)
    stk_fit = time.time() - t0

    stk_pred = stacking.predict(X_test)
    stk_proba = stacking.predict_proba(X_test)[:, 1]
    stk_metrics = {
        "name":     "test_stacking",
        "f1":        f1_score(y_test, stk_pred),
        "auc":       roc_auc_score(y_test, stk_proba),
        "accuracy":  accuracy_score(y_test, stk_pred),
        "precision": precision_score(y_test, stk_pred),
        "recall":    recall_score(y_test, stk_pred),
        "fit_time":  stk_fit,
        "infer_time": 0.0,
    }
    test_results = pd.concat([test_results, pd.DataFrame([stk_metrics]).set_index("name")])
    print(f"  Stacking fit time = {stk_fit:.1f}s")
    print(f"  Stacking on test: F1 = {stk_metrics['f1']:.4f}, AUC = {stk_metrics['auc']:.4f}")
    display(test_results.round(4))
else:
    print("Skipped stacking demo.")
"""))

# =============================================================================
# Section 7 — Save & Summary
# =============================================================================
cells.append(md(r"""
## Section 7 — Lưu artifacts & Tóm tắt

Ghi toàn bộ bảng metrics vào CSV để dễ chèn vào báo cáo Word.
"""))

cells.append(code(r"""
# Tổng hợp tất cả kết quả vào 1 DataFrame thống nhất
cv_results = pd.DataFrame([
    res_baseline_rf, res_baseline_svm,
    res_champ_rf,    res_champ_svm,
]).set_index("name")

# Ghi 2 file: cv_results và test_results (định dạng khác nhau)
cv_results.round(6).to_csv(DATA / "model_comparison_cv.csv")
test_results.round(6).to_csv(DATA / "model_comparison_test.csv")

print(f"Saved → {DATA / 'model_comparison_cv.csv'}")
print(f"Saved → {DATA / 'model_comparison_test.csv'}")
print(f"Saved → {DATA / 'best_svm_params.json'}")
print(f"Figures dir: {FIG}")
for fp in sorted(FIG.glob("03_*.png")):
    print(f"  {fp.name}")
"""))

cells.append(code(r"""
# Summary cuối — winner trên test set
test_winner = "RF" if test_results.loc["test_rf", "f1"] >= test_results.loc["test_svm", "f1"] else "SVM"

print("=" * 60)
print("KẾT LUẬN")
print("=" * 60)
print(f"Best RF params  : {rf_best_params}")
print(f"Best SVM params : {svm_best_params}")
print()
print("CV (5-Fold trên train) — F1 ± std:")
print(f"  baseline_rf  : {res_baseline_rf['f1_mean']:.4f} ± {res_baseline_rf['f1_std']:.4f}")
print(f"  baseline_svm : {res_baseline_svm['f1_mean']:.4f} ± {res_baseline_svm['f1_std']:.4f}")
print(f"  champion_rf  : {res_champ_rf['f1_mean']:.4f} ± {res_champ_rf['f1_std']:.4f}")
print(f"  champion_svm : {res_champ_svm['f1_mean']:.4f} ± {res_champ_svm['f1_std']:.4f}")
print()
print("Test set (20% holdout) — F1 / AUC / Accuracy:")
for idx, row in test_results.iterrows():
    print(f"  {idx:14s}: F1={row['f1']:.4f}  AUC={row['auc']:.4f}  Acc={row['accuracy']:.4f}  "
          f"fit={row['fit_time']:.1f}s")
print()
print(f">>> Winner trên test set (theo F1): {test_winner} <<<")
"""))

# -----------------------------------------------------------------------------
# Build & write
# -----------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out_path = Path(__file__).parent / "notebooks" / "03_rf_svm_comparison.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path}")
print(f"Total cells: {len(cells)}  "
      f"({sum(1 for c in cells if c.cell_type=='markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type=='code')} code)")
