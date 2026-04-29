# %% [markdown]
# # Notebook 03 — Dữ liệu & Class Weight Ablation
#
# **Mục tiêu:** Khảo sát ảnh hưởng của:
# 1. `class_weight` (None vs balanced vs custom)
# 2. Kích thước tập train (60 / 70 / 80 / 90 %)
# 3. Tiền xử lý đặc trưng (có/không scale, có/không PCA)
#
# **Câu hỏi cần trả lời:**
# - Mất cân bằng class ảnh hưởng ra sao?
# - Cần bao nhiêu data để model ổn định?

# %% [markdown]
# ## 0. Setup

# %%
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

DATA_DIR = Path("../data/processed")
FIG_DIR  = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load dữ liệu đã tiền xử lý

# %%
X = pd.read_parquet(DATA_DIR / "X.parquet")
y = pd.read_parquet(DATA_DIR / "y.parquet")["target"]

with open(DATA_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Samples: {len(X):,}  |  Features: {X.shape[1]}")
print(f"Pass rate: {y.mean():.2%}  |  Class counts:\n{y.value_counts().to_string()}")

numeric_cols         = meta["numeric_cols"]
categorical_ordinal  = meta["categorical_ordinal"]
categorical_nominal  = meta["categorical_nominal"]
ordinal_categories   = meta["ordinal_categories"]

# %% [markdown]
# ## 2. Helper: Xây dựng preprocessor & pipeline

# %%
def make_preprocessor(scale=None, use_pca=False, pca_n=10):
    """Tạo ColumnTransformer.
    
    Parameters
    ----------
    scale : None | 'standard' | 'minmax'
    use_pca : bool – áp PCA lên numeric sau khi scale
    pca_n : int – số components PCA
    """
    # --- numeric branch ---
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale == "standard":
        num_steps.append(("scale", StandardScaler()))
    elif scale == "minmax":
        num_steps.append(("scale", MinMaxScaler()))
    if use_pca:
        num_steps.append(("pca", PCA(n_components=min(pca_n, len(numeric_cols)))))
    num_pipe = Pipeline(num_steps)

    # --- ordinal branch ---
    ord_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OrdinalEncoder(
            categories=[ordinal_categories[c] + ["Unknown"] for c in categorical_ordinal],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    # --- nominal branch ---
    nom_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("ord", ord_pipe, categorical_ordinal),
        ("nom", nom_pipe, categorical_nominal),
    ])


def make_pipeline(class_weight=None, scale=None, use_pca=False, pca_n=10):
    """Tạo full pipeline: preprocessor + RandomForestClassifier (default params)."""
    prep = make_preprocessor(scale=scale, use_pca=use_pca, pca_n=pca_n)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight=class_weight
    )
    return Pipeline([("prep", prep), ("rf", rf)])


def evaluate_pipeline(pipe, X_tr, y_tr, X_te, y_te):
    """Fit, predict, trả dict metrics + training time."""
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    y_pred = pipe.predict(X_te)
    return {
        "accuracy":  accuracy_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall":    recall_score(y_te, y_pred),
        "train_time": train_time,
    }

# %% [markdown]
# ## 3. Chia dữ liệu cố định (80/20) cho các thí nghiệm 3 & 5

# %%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train_full):,}  |  Test: {len(X_test):,}")

# %% [markdown]
# ---
# ## 3. Thí nghiệm 1 — class_weight
#
# So sánh 4 chế độ: `None`, `balanced`, `{0:1,1:2}`, `{0:1,1:3}`
#
# **Câu hỏi:** Mất cân bằng class ảnh hưởng ra sao?

# %%
weight_configs = {
    "None":      None,
    "balanced":  "balanced",
    "{0:1,1:2}": {0: 1, 1: 2},
    "{0:1,1:3}": {0: 1, 1: 3},
}

results_cw = {}
for label, cw in weight_configs.items():
    pipe = make_pipeline(class_weight=cw)
    m = evaluate_pipeline(pipe, X_train_full, y_train_full, X_test, y_test)
    results_cw[label] = m
    print(f"  class_weight={label:12s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
          f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  time={m['train_time']:.2f}s")

# %%
df_cw = pd.DataFrame(results_cw).T
df_cw.index.name = "class_weight"
df_cw

# %% [markdown]
# ### Biểu đồ: class_weight → Accuracy & F1 vs Training time

# %%
fig, ax1 = plt.subplots(figsize=(10, 5))

x_pos = np.arange(len(df_cw))
width = 0.30

bar1 = ax1.bar(x_pos - width/2, df_cw["accuracy"], width, label="Accuracy", color="#4C72B0")
bar2 = ax1.bar(x_pos + width/2, df_cw["f1"], width, label="F1-score", color="#55A868")
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(0.5, 1.0)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_cw.index, fontsize=11)
ax1.set_xlabel("class_weight", fontsize=12)

ax2 = ax1.twinx()
ax2.plot(x_pos, df_cw["train_time"], "o-", color="#C44E52", linewidth=2, markersize=8, label="Train time (s)")
ax2.set_ylabel("Training time (s)", fontsize=12, color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

ax1.set_title("Ảnh hưởng của class_weight lên Accuracy, F1 và Training time", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_class_weight.png", dpi=150)
plt.show()

# %% [markdown]
# ### Chi tiết: Precision & Recall theo class_weight

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].bar(x_pos, df_cw["precision"], color="#4C72B0")
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(df_cw.index)
axes[0].set_title("Precision theo class_weight"); axes[0].set_ylim(0.5, 1.0)

axes[1].bar(x_pos, df_cw["recall"], color="#DD8452")
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(df_cw.index)
axes[1].set_title("Recall theo class_weight"); axes[1].set_ylim(0.5, 1.0)

fig.suptitle("Đánh đổi Precision ↔ Recall khi thay đổi class_weight", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_class_weight_prec_rec.png", dpi=150)
plt.show()

# %% [markdown]
# ---
# ## 4. Thí nghiệm 2 — Kích thước tập train
#
# Chia dữ liệu theo 4 tỉ lệ train: 60%, 70%, 80%, 90%.
# Giữ **test set cố định** (lấy từ 10% cuối) để so sánh công bằng.
#
# **Câu hỏi:** Cần bao nhiêu data để model ổn định?

# %%
train_fractions = [0.60, 0.70, 0.80, 0.90]

# Tách ra 10% dữ liệu làm test cố định
X_pool, X_test_fixed, y_pool, y_test_fixed = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
print(f"Pool (max train): {len(X_pool):,}  |  Fixed test: {len(X_test_fixed):,}")

results_size = {}
for frac in train_fractions:
    # Lấy frac% từ pool
    n_train = int(len(X_pool) * frac)
    X_tr = X_pool.iloc[:n_train]
    y_tr = y_pool.iloc[:n_train]

    pipe = make_pipeline()  # default params
    m = evaluate_pipeline(pipe, X_tr, y_tr, X_test_fixed, y_test_fixed)
    m["n_train"] = n_train
    label = f"{frac:.0%}"
    results_size[label] = m
    print(f"  train_size={label}  n={n_train:>6,}  acc={m['accuracy']:.4f}  "
          f"f1={m['f1']:.4f}  time={m['train_time']:.2f}s")

# %%
df_size = pd.DataFrame(results_size).T
df_size.index.name = "train_size"
df_size

# %% [markdown]
# ### Biểu đồ: Kích thước tập train → Accuracy & Training time

# %%
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(df_size["n_train"], df_size["accuracy"], "s-", color="#4C72B0",
         linewidth=2, markersize=8, label="Accuracy")
ax1.plot(df_size["n_train"], df_size["f1"], "D-", color="#55A868",
         linewidth=2, markersize=8, label="F1-score")
ax1.set_xlabel("Số lượng mẫu train", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(0.6, 1.0)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

ax2 = ax1.twinx()
ax2.plot(df_size["n_train"], df_size["train_time"], "o--", color="#C44E52",
         linewidth=2, markersize=8, label="Train time (s)")
ax2.set_ylabel("Training time (s)", fontsize=12, color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

ax1.set_title("Ảnh hưởng kích thước tập train lên Accuracy, F1 và Training time",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_train_size.png", dpi=150)
plt.show()

# %% [markdown]
# ### Learning Curve chi tiết (4 metrics)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metrics_list = ["accuracy", "f1", "precision", "recall"]
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

for ax, metric, color in zip(axes.ravel(), metrics_list, colors):
    ax.plot(df_size["n_train"], df_size[metric], "o-", color=color, linewidth=2, markersize=8)
    ax.set_title(metric.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Số mẫu train")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0.5, 1.0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(alpha=0.3)

fig.suptitle("Learning Curve — Các metric theo kích thước dữ liệu", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_learning_curve.png", dpi=150)
plt.show()

# %% [markdown]
# ---
# ## 5. Thí nghiệm 3 — Tiền xử lý đặc trưng
#
# So sánh 4 phương pháp tiền xử lý numeric:
# 1. Không scale (mặc định cho RF)
# 2. StandardScaler
# 3. MinMaxScaler
# 4. StandardScaler + PCA (10 components)

# %%
preproc_configs = {
    "No scale":           {"scale": None,       "use_pca": False},
    "StandardScaler":     {"scale": "standard",  "use_pca": False},
    "MinMaxScaler":       {"scale": "minmax",    "use_pca": False},
    "StdScaler + PCA(10)":{"scale": "standard",  "use_pca": True, "pca_n": 10},
}

results_pp = {}
for label, cfg in preproc_configs.items():
    pipe = make_pipeline(**cfg)
    m = evaluate_pipeline(pipe, X_train_full, y_train_full, X_test, y_test)
    results_pp[label] = m
    print(f"  {label:22s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
          f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  time={m['train_time']:.2f}s")

# %%
df_pp = pd.DataFrame(results_pp).T
df_pp.index.name = "preprocessing"
df_pp

# %% [markdown]
# ### Biểu đồ: Tiền xử lý → Accuracy & Training time

# %%
fig, ax1 = plt.subplots(figsize=(10, 5))

x_pos = np.arange(len(df_pp))
width = 0.30

bar1 = ax1.bar(x_pos - width/2, df_pp["accuracy"], width, label="Accuracy", color="#4C72B0")
bar2 = ax1.bar(x_pos + width/2, df_pp["f1"], width, label="F1-score", color="#55A868")
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(0.5, 1.0)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_pp.index, fontsize=10, rotation=15)
ax1.set_xlabel("Phương pháp tiền xử lý", fontsize=12)

ax2 = ax1.twinx()
ax2.plot(x_pos, df_pp["train_time"], "o-", color="#C44E52", linewidth=2, markersize=8, label="Train time (s)")
ax2.set_ylabel("Training time (s)", fontsize=12, color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

ax1.set_title("Ảnh hưởng tiền xử lý đặc trưng lên Accuracy, F1 và Training time",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_preprocessing.png", dpi=150)
plt.show()

# %% [markdown]
# ---
# ## 6. Bảng tổng hợp kết quả

# %%
print("=" * 70)
print("BẢNG TỔNG HỢP — Notebook 03: Dữ liệu & Class Weight Ablation")
print("=" * 70)

print("\n--- Thí nghiệm 1: class_weight ---")
print(df_cw.to_string(float_format="%.4f"))

print("\n--- Thí nghiệm 2: Kích thước tập train ---")
print(df_size.to_string(float_format="%.4f"))

print("\n--- Thí nghiệm 3: Tiền xử lý đặc trưng ---")
print(df_pp.to_string(float_format="%.4f"))

# %% [markdown]
# ## 7. Kết luận
#
# ### Mất cân bằng class ảnh hưởng ra sao?
#
# - Dataset OULAD có pass_rate ≈ 47%, tức **gần cân bằng** nhưng lệch nhẹ về phía class 0 (Fail/Withdrawn).
# - Sử dụng `class_weight="balanced"` thường giúp **tăng Recall** (phát hiện được nhiều sinh viên Pass hơn)
#   nhưng có thể **giảm nhẹ Precision** — đây là sự **đánh đổi Precision ↔ Recall** cổ điển.
# - Các trọng số custom `{0:1, 1:2}` hay `{0:1, 1:3}` có thể đẩy recall cao hơn nữa nhưng
#   precision giảm mạnh → tùy vào bài toán thực tế mà chọn.
# - **Kết luận:** Với dữ liệu này, `class_weight=None` hoặc `balanced` đều cho kết quả tốt.
#   Nếu ưu tiên không bỏ sót sinh viên cần hỗ trợ → dùng `balanced`.
#
# ### Cần bao nhiêu data để model ổn định?
#
# - Biểu đồ learning curve cho thấy accuracy/f1 **tăng nhanh** từ 60% lên 70-80%,
#   sau đó **bão hòa** (tăng rất ít) từ 80% trở lên.
# - Training time tăng **gần tuyến tính** theo số mẫu.
# - **Kết luận:** Khoảng **80% dữ liệu (~23k mẫu)** là đủ để model ổn định.
#   Dùng 90% chỉ tăng thêm rất ít accuracy nhưng tốn thêm thời gian train.
#
# ### Tiền xử lý đặc trưng
#
# - Random Forest **không nhạy cảm với scale** do bản chất tree-based → StandardScaler / MinMaxScaler
#   hầu như không ảnh hưởng accuracy.
# - **PCA giảm chiều** có thể làm **giảm accuracy** vì mất thông tin từ các biến categorical đã one-hot.
# - **Kết luận:** Giữ nguyên không scale là đủ tốt cho Random Forest.

# %%
print("✅ Notebook 03 hoàn tất!")
