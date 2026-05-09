# %% [markdown]
# # Vẽ biểu đồ Learning Curve cho mô hình Random Forest
# 
# Quy trình:
# 1. Load dữ liệu đã xử lý (`merged_student_data.csv`).
# 2. Sử dụng `learning_curve` từ sklearn để tính toán độ chính xác theo kích thước tập mẫu.
# 3. Trực quan hóa kết quả hệt như hình mẫu.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import os

# Cấu hình đường dẫn
PROCESSED_FILE = '../data/processed/merged_student_data.csv'
REPORT_DIR = '../reports/figures/'
os.makedirs(REPORT_DIR, exist_ok=True)

# %%
# 1. Load Data
data = pd.read_csv(PROCESSED_FILE)
X = data.drop(columns=['target'])
y = data['target']

# %%
# 2. Tính toán Learning Curve
print("Calculating learning curve... This might take a moment.")
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    X, y, 
    cv=5, 
    n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Tính trung bình và độ lệch chuẩn
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# %%
# 3. Trực quan hóa (Theo phong cách hình mẫu)
plt.figure(figsize=(10, 6))
plt.title("Learning Curve (Random Forest)")
plt.xlabel("Sample Size (Training examples)")
plt.ylabel("Accuracy")
plt.grid()

# Vẽ dải độ lệch chuẩn
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

# Vẽ đường trung bình
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.tight_layout()

# Lưu biểu đồ
output_path = os.path.join(REPORT_DIR, 'learning_curve.png')
plt.savefig(output_path)
print(f"Learning curve saved to: {output_path}")
# plt.show()
