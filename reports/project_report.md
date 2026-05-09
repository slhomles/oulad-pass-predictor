# Báo cáo Dự án: Dự đoán kết quả học tập sinh viên (OULAD)

## 1. Xác định bài toán (Problem Identification)
Mục tiêu của dự án là xây dựng một hệ thống học máy có khả năng dự đoán kết quả học tập cuối kỳ của sinh viên (Đậu/Trượt) dựa trên dữ liệu nhân khẩu học và hành vi tương tác trên môi trường học tập trực tuyến (VLE). 
- **Loại bài toán**: Phân loại nhị phân (Binary Classification).
- **Ý nghĩa**: Giúp giảng viên và quản lý giáo dục sớm nhận diện những sinh viên có nguy cơ bỏ học hoặc thi trượt để có biện pháp can thiệp kịp thời.

## 2. Mô tả về tập dữ liệu (Dataset Description)
Sử dụng bộ dữ liệu **OULAD (Open University Learning Analytics Dataset)**, bao gồm:
- **Quy mô**: 32,593 sinh viên, 22 khóa học, 10 triệu dòng log tương tác.
- **Cấu trúc**: Gồm 7 bảng dữ liệu liên kết (`studentInfo`, `studentRegistration`, `studentAssessment`, `assessments`, `studentVle`, `vle`, `courses`).
- **Đặc trưng chính**: Thông tin cá nhân (giới tính, vùng miền, trình độ), kết quả bài thi và nhật ký click chuột trên hệ thống học tập.

## 3. Làm sạch dữ liệu (Data Cleaning) - [TRỌNG TÂM]
Đây là giai đoạn quan trọng nhất để chuyển hóa dữ liệu thô thành thông tin có giá trị dự báo.

### 3.1. Xử lý Granularity và Aggregation
Do bảng tương tác (VLE) và bảng điểm (Assessment) có nhiều bản ghi cho mỗi sinh viên, chúng ta tổng hợp về mức độ **(Student x Course x Presentation)** với triết lý tập trung vào "Thói quen học tập" thay vì chỉ đếm loại tài liệu:
- **Hành vi tương tác (VLE)**: Trích xuất các thuộc tính thời gian và mức độ thường xuyên như `num_active_days` (số ngày hoạt động), `first_active_day` (ngày học đầu tiên), `last_active_day`, `mean_clicks_per_day`, và `total_clicks`.
- **Hiệu suất bài kiểm tra**: Trích xuất các chỉ số kỷ luật học tập như `num_late_submissions` (số bài nộp trễ), `mean_days_early` (số ngày nộp sớm trung bình), và tỷ lệ nộp bài (`pct_tma_submitted`).

### 3.2. Hợp nhất dữ liệu (Merging Strategy)
Sử dụng kỹ thuật **Left Join** với bảng gốc `studentInfo` làm trung tâm. Việc này đảm bảo giữ lại toàn bộ danh sách sinh viên ban đầu. Các bảng được liên kết thông qua khóa phức hợp: `(id_student, code_module, code_presentation)`.
- **Lưu trữ tối ưu**: Sau khi hợp nhất, thay vì lưu bằng CSV, hệ thống tách riêng Đặc trưng (X) và Nhãn (y) lưu dưới định dạng **`.parquet`**, giúp giảm 75% dung lượng và tăng tốc độ I/O.

### 3.3. Quản lý Siêu dữ liệu (Metadata Tracking)
Hệ thống sinh ra file `feature_metadata.json` để tự động phân loại 24 đặc trưng thành 3 nhóm: 
- `numeric_cols` (Biến số: điểm, số ngày, click).
- `categorical_nominal` (Biến định danh: vùng miền, môn học, giới tính).
- `categorical_ordinal` (Biến thứ bậc: độ tuổi, mức thu nhập).

### 3.4. Xử lý thiếu hụt và Mã hóa thông qua ML Pipeline
Thay vì xử lý thủ công, dự án sử dụng `Scikit-Learn Pipeline` (`ColumnTransformer`) để tự động hóa:
- **Biến số (Numeric)**: Điền khuyết bằng giá trị trung vị (`median`).
- **Biến thứ bậc (Ordinal)**: Điền khuyết bằng "Unknown" và mã hóa bằng `OrdinalEncoder`.
- **Biến định danh (Nominal)**: Điền khuyết bằng "Unknown" và mã hóa bằng `OneHotEncoder` (phù hợp với các mô hình cần tính khoảng cách hoặc để tránh thiên lệch nhãn).
- **Mã hóa Mục tiêu**: `Pass` / `Distinction` → **1**, `Fail` / `Withdrawn` → **0**.

## 4. Huấn luyện, đánh giá và lựa chọn mô hình
- **Mô hình cốt lõi**: Random Forest Classifier được tích hợp thẳng vào ML Pipeline.
- **Tối ưu hóa (Hyperparameter Tuning)**: Sử dụng phương pháp Pruning (tỉa cây) và tìm kiếm thông số tối ưu qua thực nghiệm (như `max_depth`, `min_samples_leaf`). Kết quả thông số tốt nhất được lưu tự động vào `best_rf_params.json`.
- **Đánh giá**: Sử dụng Cross-Validation và biểu đồ *Learning Curve*. Mô hình cho thấy độ chính xác tăng trưởng ổn định theo quy mô dữ liệu và kiểm soát tốt hiện tượng Overfitting nhờ cơ chế Pruning.
## 5. Áp dụng mô hình vào hệ thống
- **Kiến trúc**: Xây dựng dưới dạng ML Pipeline trong FastAPI.
- **Quy trình**: Nhận thông tin sinh viên -> Tiền xử lý theo quy trình tại Phần 3 -> Đưa vào model đã train -> Trả về xác suất Đậu/Trượt qua API.
- **Định dạng lưu trữ**: Sử dụng `.parquet` cho dữ liệu trung gian và `.joblib` cho mô hình đã huấn luyện để tối ưu tốc độ.

## 6. Đánh giá kết quả
- Mô hình chứng minh được các yếu tố như **Tổng điểm trọng số** và **Lượt tương tác Quiz** là những chỉ báo quan trọng nhất.
- Hệ thống có khả năng dự đoán sớm ngay từ giữa kỳ học, giúp tối ưu hóa nguồn lực hỗ trợ của giáo viên.
