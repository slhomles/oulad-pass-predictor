# %% [markdown]
# # Khám phá và Xử lý dữ liệu OULAD (Phiên bản Nâng cao - 20 Thuộc tính)
#
# Quy trình trong notebook này:
# 1. Đọc dữ liệu từ tất cả các bảng (.csv).
# 2. **Feature Engineering**:
#    - Đăng ký: Thời gian đăng ký + trạng thái hủy (`is_withdrawn`).
#    - Đánh giá: Điểm trọng số, độ trễ nộp bài, số lượng bài đã nộp.
#    - Khóa học: Thời lượng module (`module_presentation_length`).
#    - VLE: Tổng click, số tài liệu duy nhất, và số click chia theo từng loại hoạt động (Pivot).
# 3. Hợp nhất thành 1 bảng duy nhất.
# 4. Chạy Random Forest để chọn ra **Top 20** thuộc tính quan trọng nhất.
# 5. Lưu bảng dữ liệu chuẩn hóa cuối cùng.

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
# Nạp toàn bộ 7 bảng dữ liệu để khai thác tối đa thông tin.

# %%
print("Loading all datasets...")
student_info = pd.read_csv(RAW_DIR + 'studentInfo.csv')
student_vle = pd.read_csv(RAW_DIR + 'studentVle.csv')
student_assessment = pd.read_csv(RAW_DIR + 'studentAssessment.csv')
assessments = pd.read_csv(RAW_DIR + 'assessments.csv')
vle = pd.read_csv(RAW_DIR + 'vle.csv')
courses = pd.read_csv(RAW_DIR + 'courses.csv')
registration = pd.read_csv(RAW_DIR + 'studentRegistration.csv')

print("Data loaded successfully.")

# %% [markdown]
# ## 1.5. Khám phá sơ bộ (EDA Nhanh)
#
# Kiểm tra kích thước dữ liệu và phân bố nhãn trước khi feature engineering.

# %%
print("=== Tổng quan dữ liệu ===")
print(f"studentInfo:       {student_info.shape[0]:,} rows × {student_info.shape[1]} cols")
print(f"studentVle:        {student_vle.shape[0]:,} rows × {student_vle.shape[1]} cols")
print(f"studentAssessment: {student_assessment.shape[0]:,} rows × {student_assessment.shape[1]} cols")
print(f"vle (activities):  {vle['activity_type'].nunique()} unique activity types")

print("\n=== Phân bố kết quả (final_result) ===")
result_counts = student_info['final_result'].value_counts()
print(result_counts)

plt.figure(figsize=(7, 4))
colors = ['#27ae60', '#2ecc71', '#e74c3c', '#c0392b']
result_counts.plot(kind='bar', color=colors, edgecolor='white')
plt.title("Phân bố kết quả học (final_result)")
plt.xlabel("")
plt.ylabel("Số sinh viên")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORT_DIR + 'class_distribution.png')
plt.show()

pass_rate = student_info['final_result'].isin(['Pass', 'Distinction']).mean()
print(f"\nTỉ lệ Pass+Distinction: {pass_rate:.2%}")

# %% [markdown]
# ## 2. Feature Engineering - Trích xuất đặc trưng chuyên sâu

# %% [markdown]
# ### 2.1. Nhóm Đăng ký (Registration)

# %%
print("Processing Registration data...")
reg_feat = registration[['id_student', 'code_module', 'code_presentation',
                          'date_registration', 'date_unregistration']].copy()
reg_feat['date_registration'] = reg_feat['date_registration'].fillna(0)
# Sinh viên hủy đăng ký giữa chừng — dấu hiệu Withdrawn quan trọng
reg_feat['is_withdrawn'] = reg_feat['date_unregistration'].notna().astype(int)
reg_feat = reg_feat.drop(columns=['date_unregistration'])

# %% [markdown]
# ### 2.2. Nhóm Đánh giá (Assessments)

# %%
print("Processing Assessment performance...")
# Merge điểm với thông tin trọng số và ngày hạn định
st_assess = student_assessment.merge(assessments, on='id_assessment', how='left')

# Tính độ trễ nộp bài (âm là nộp sớm, dương là nộp muộn)
st_assess['submission_delay'] = st_assess['date_submitted'] - st_assess['date']
# Tính điểm có trọng số
st_assess['weighted_score'] = st_assess['score'] * st_assess['weight'] / 100

# Aggregation theo từng sinh viên trong mỗi khóa học
assess_agg = st_assess.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    avg_score=('score', 'mean'),
    avg_submission_delay=('submission_delay', 'mean'),
    total_weighted_score=('weighted_score', 'sum'),
    assessment_count=('id_assessment', 'count')
).reset_index()

# %% [markdown]
# ### 2.3. Nhóm Khóa học (Courses)
#
# Lấy `module_presentation_length` — thời lượng khóa học (ngày).
# Đây là thông tin cấp module, không phải cấp sinh viên.

# %%
print("Processing Courses data...")
course_feat = courses[['code_module', 'code_presentation', 'module_presentation_length']].copy()
print(f"module_presentation_length: min={course_feat['module_presentation_length'].min()}, "
      f"max={course_feat['module_presentation_length'].max()} ngày")

# %% [markdown]
# ### 2.4. Nhóm Hoạt động VLE (VLE Activity)

# %%
print("Processing VLE interaction types...")
# Merge log tương tác với loại tài liệu
st_vle = student_vle.merge(vle[['id_site', 'activity_type']], on='id_site', how='left')

# Tổng số click và số lượng site duy nhất
vle_basic = st_vle.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    total_vle_clicks=('sum_click', 'sum'),
    unique_sites_visited=('id_site', 'nunique')
).reset_index()

# Pivot: số click theo từng loại hoạt động
vle_pivot = st_vle.pivot_table(
    index=['id_student', 'code_module', 'code_presentation'],
    columns='activity_type',
    values='sum_click',
    aggfunc='sum',
    fill_value=0
).reset_index()
# Xóa tên axis do pivot_table tạo ra để tránh xung đột khi merge
vle_pivot.columns.name = None

vle_feat = vle_basic.merge(vle_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')
print(f"VLE features: {vle_feat.shape[1] - 3} cột  |  Activity types: {vle_feat.shape[1] - 5}")

# %% [markdown]
# ## 3. Merging - Hợp nhất toàn bộ

# %%
print("Merging all features into one master table...")
# Gốc là student_info
master_df = student_info.merge(reg_feat, on=['id_student', 'code_module', 'code_presentation'], how='left')
master_df = master_df.merge(assess_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')
master_df = master_df.merge(vle_feat, on=['id_student', 'code_module', 'code_presentation'], how='left')
master_df = master_df.merge(course_feat, on=['code_module', 'code_presentation'], how='left')

# Điền các giá trị thiếu (sv không có hoạt động VLE hoặc bài nộp)
master_df = master_df.fillna(0)

n_feature_cols = master_df.shape[1] - 4  # trừ id_student, code_module, code_presentation, final_result
print(f"Master Table Shape: {master_df.shape}")
print(f"Tổng số features trước khi chọn: {n_feature_cols}")

# %% [markdown]
# ## 4. Encoding & Preparation

# %%
print("Encoding categorical variables...")
# Target: Pass/Distinction = 1, Fail/Withdrawn = 0
target_map = {'Pass': 1, 'Distinction': 1, 'Fail': 0, 'Withdrawn': 0}
master_df['target'] = master_df['final_result'].map(target_map)

# Features
X_raw = master_df.drop(columns=['final_result', 'target', 'id_student', 'code_module', 'code_presentation'])
y = master_df['target']

# Label Encoding cho các cột dạng object
le = LabelEncoder()
for col in X_raw.select_dtypes(include=['object']).columns:
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

# %% [markdown]
# ## 5. Feature Selection (Top 20)

# %%
print("Running Random Forest to select Top 20 features...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_raw, y)

# Lấy độ quan trọng
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
all_features = X_raw.columns

# Chọn Top 20
top_20_indices = indices[:20]
top_20_features = [all_features[i] for i in top_20_indices]

print("\nTop 20 Most Valuable Features:")
for i, feat in enumerate(top_20_features):
    print(f"{i+1}. {feat} ({importances[indices[i]]:.4f})")

# Trực quan hóa
plt.figure(figsize=(12, 8))
plt.title("Top 20 Feature Importances (OULAD)")
sns.barplot(x=importances[top_20_indices], y=top_20_features, palette="magma")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(REPORT_DIR + 'feature_importances.png')
# plt.show()

# %% [markdown]
# ## 5.5. Phân tích tương quan (Top 20 Features)
#
# Heatmap tương quan giúp phát hiện multicollinearity giữa các features đã chọn.

# %%
# Heatmap tương quan của Top 20 features với target
corr_data = X_raw[top_20_features].copy()
corr_data['target'] = y
corr_matrix = corr_data.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            linewidths=0.3, square=False)
plt.title("Correlation Heatmap — Top 20 Features + Target")
plt.tight_layout()
plt.savefig(REPORT_DIR + 'correlation_heatmap.png')
plt.show()

# %% [markdown]
# ## 6. Lưu kết quả

# %%
final_dataset = X_raw[top_20_features].copy()
final_dataset['target'] = y

assert len(top_20_features) == 20, f"Lỗi: chỉ có {len(top_20_features)} features thay vì 20!"

output_path = PROCESSED_DIR + 'merged_student_data.csv'
final_dataset.to_csv(output_path, index=False)

print(f"=== Kết quả cuối cùng ===")
print(f"Số thuộc tính: {len(top_20_features)}")
print(f"Tổng mẫu:      {len(final_dataset):,}")
print(f"Tỉ lệ Pass+Distinction (target=1): {y.mean():.2%}")
print(f"\nDanh sách 20 thuộc tính:")
for i, col in enumerate(top_20_features, 1):
    print(f"  {i:2d}. {col}")
print(f"\nĐã lưu vào: {output_path}")
print("Done!")
