# %% [markdown]
# # Khám phá và Xử lý dữ liệu OULAD
# 
# Quy trình trong notebook này:
# 1. Đọc dữ liệu thô từ `data/raw/`
# 2. Tổng hợp (Aggregate) điểm số các bài kiểm tra (từ bảng `studentAssessment`) và số lượt tương tác VLE (bảng `studentVle`).
# 3. Hợp nhất (Merge) các bảng lại thành 1 bảng suy nhất (Base: `studentInfo.csv`).
# 4. Chạy mô hình Random Forest để đánh giá Độ quan trọng của các biến (Feature Importance).
# 5. Lưu bảng dữ liệu đã chọn lọc (làm sạch + chuẩn hóa) phục vụ huấn luyện.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Cấu hình đường dẫn
RAW_DIR = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
REPORT_DIR = '../reports/figures/'

# Khởi tạo thư mục nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Data
# Đọc các bảng quan trọng nhất từ thư mục dữ liệu thô.

# %%
print("Loading data...")
student_info = pd.read_csv(RAW_DIR + 'studentInfo.csv')
student_vle = pd.read_csv(RAW_DIR + 'studentVle.csv')
student_assessment = pd.read_csv(RAW_DIR + 'studentAssessment.csv')

print(f"Student Info: {student_info.shape}")
print(f"Student VLE: {student_vle.shape}")
print(f"Student Assessment: {student_assessment.shape}")

# %% [markdown]
# ## 2. Aggregation - Tổng hợp dữ liệu
# Vì 1 học sinh tương tác VLE rất nhiều lần và làm nhiều bài thi trong khóa học, 
# ta cần gom (Group By) thông tin này.

# %%
print("Aggregating VLE and Assessments...")
# VLE: Tổng số click của từng sinh viên trong mỗi môn học
vle_agg = student_vle.groupby(['id_student', 'code_module', 'code_presentation'])['sum_click'].sum().reset_index()
vle_agg.rename(columns={'sum_click': 'total_vle_clicks'}, inplace=True)

# Assessment: Điểm trung bình các bài kiểm tra của sinh viên
assessment_agg = student_assessment.groupby('id_student')['score'].mean().reset_index()
assessment_agg.rename(columns={'score': 'avg_assessment_score'}, inplace=True)

# %% [markdown]
# ## 3. Merging - Ghép bảng

# %%
print("Merging tables...")
# Bảng căn bản
merged_df = student_info.merge(vle_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')

# Ghép tiếp điểm bài kiểm tra
merged_df = merged_df.merge(assessment_agg, on='id_student', how='left')

# Imputation: Điền giá trị 0 cho các học sinh không click VLE hoặc không có điểm kiểm tra
# Việc sinh viên không thi có thể hiểu là không có điểm (0), không click là độ tương tác bằng 0
merged_df['total_vle_clicks'] = merged_df['total_vle_clicks'].fillna(0)
merged_df['avg_assessment_score'] = merged_df['avg_assessment_score'].fillna(0)

print("Merged Data Shape:", merged_df.shape)
merged_df.head()

# %% [markdown]
# ## 4. Encoding - Chuẩn hóa biến hạng mục

# %%
print("Encoding variables...")
# Xử lý biến Target (final_result)
# Gộp về bài toán phân loại nhị phân: Pass/Distinction = 1, Fail/Withdrawn = 0
target_map = {'Pass': 1, 'Distinction': 1, 'Fail': 0, 'Withdrawn': 0}
merged_df['final_result_encoded'] = merged_df['final_result'].map(target_map)

# Tạo tập Features đầu vào
features = merged_df.drop(columns=['final_result', 'final_result_encoded', 'id_student', 'code_module', 'code_presentation'])

# Categorical Encoding (Sử dụng Label Encoding để giữ DataFrame gọn gọn, phù hợp cho Random Forest)
categorical_cols = features.select_dtypes(include=['object']).columns
for col in categorical_cols:
    features[col] = features[col].astype(str)
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

X = features.fillna(0)
y = merged_df['final_result_encoded']

print("Features Shape:", X.shape)

# %% [markdown]
# ## 5. Lựa chọn thuộc tính với Random Forest (Feature Selection)
# Ta huấn luyện tập X với thuật toán RF để đánh giá mức độ đóng góp (Quan trọng) của từng biến.

# %%
print("Fitting Random Forest to extract Feature Importances...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Visualization
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Độ Quan Trọng Của Các Thuộc Tính (Feature Importances)")
sns.barplot(x=importances[indices], y=names, palette="viridis")
plt.xlabel("Tỷ lệ phần trăm đóng góp")
plt.tight_layout()

# Lưu biểu đồ
fig_path = REPORT_DIR + 'feature_importances.png'
plt.savefig(fig_path)
print(f"Saved feature importance plot to: {fig_path}")

# plt.show()

# %% [markdown]
# ## 6. Tạo đầu ra (Lọc thuộc tính và Lưu file)
# Giải thích cho câu hỏi: "Thế nào là cần thiết?"
# Thông qua bảng RandomForest Importance ở trên, những feature có đóng góp < 1% hầu như không tác động đến điểm số sinh viên. Ta có thể loại bỏ an toàn.

# %%
threshold = 0.01  # Lọc bỏ các thuộc tính đóng góp dưới 1%
selected_features = [X.columns[i] for i, imp in enumerate(importances) if imp > threshold]

print("Stats:")
print(f"Total initial features: {X.shape[1]}")
print(f"Total selected features: {len(selected_features)}")
print("Selected features:", selected_features)

# Tạo DataFrame chuẩn hóa cuối cùng
final_dataset = X[selected_features].copy()
final_dataset['target'] = y

output_path = PROCESSED_DIR + 'merged_student_data.csv'
final_dataset.to_csv(output_path, index=False)
print(f"Saved final dataset to: {output_path}")
print("Done!")
