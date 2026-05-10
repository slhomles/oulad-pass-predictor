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

### 3.4. Cấu trúc Tiền xử lý (ML Pipeline) Dùng chung & Riêng biệt
Hệ thống sử dụng `Scikit-Learn Pipeline` (`ColumnTransformer`) để tự động hóa toàn bộ luồng tiền xử lý. Quá trình này được thiết kế để chia sẻ các bước xử lý cơ sở (dùng chung cho cả 2 thuật toán) và phân tách các bước đặc thù:

**A. Các bước dùng chung (Chung cho cả Random Forest và LinearSVC)**
- **Biến số (Numeric)**: Điền khuyết bằng giá trị trung vị (`median`).
- **Biến thứ bậc (Ordinal)**: Điền khuyết bằng "Unknown" và mã hóa bằng `OrdinalEncoder`.
- **Biến định danh (Nominal)**: Điền khuyết bằng "Unknown" và mã hóa bằng `OneHotEncoder`.
- **Mã hóa Mục tiêu**: `Pass` / `Distinction` → **1**, `Fail` / `Withdrawn` → **0**.

**B. Các bước dùng riêng (Thuật toán đặc thù)**
- **Random Forest**: Không yêu cầu thêm bất kỳ bước xử lý nào sau khi điền khuyết và mã hóa, do bản chất thuật toán cây quyết định hoạt động dựa trên các ngưỡng cắt (splits) độc lập với thang đo.
- **LinearSVC (SVM) và Chiến lược Chuẩn hóa (Scaling)**: Thuật toán SVM hoạt động dựa trên khoảng cách hình học trong không gian đa chiều, do đó bắt buộc phải có cơ chế đồng bộ biên độ của dữ liệu. Hệ thống sử dụng bộ chuẩn hóa **`StandardScaler`** theo phương pháp **Z-score** (đưa giá trị trung bình về 0, độ lệch chuẩn về 1) thay vì Min-Max Scaler nhằm tăng khả năng kháng cự với các dữ liệu ngoại lai (outliers - ví dụ sinh viên click hàng vạn lần). Việc chuẩn hóa không áp dụng mù quáng trên toàn bộ dữ liệu mà được phân tách thông minh:
  - **Nhóm 19 đặc trưng (Bắt buộc qua Z-score)**: Bao gồm 16 biến số (`num_of_prev_attempts`, `studied_credits`, `total_clicks`, `num_active_days`, `max_clicks_day`, `first_active_day`, `last_active_day`, `num_resources_accessed`, `mean_clicks_per_day`, `mean_score`, `num_submissions`, `num_late_submissions`, `mean_days_early`, `pct_tma_submitted`, `pct_cma_submitted`, `registered_days_before`) và 3 biến thứ bậc đã được ép sang số (`highest_education`, `age_band`, `imd_band`). Nhóm này bắt buộc đi qua màng lọc Z-score để đưa về cùng một dải phân phối, tránh việc các biến lớn "đè bẹp" các biến nhỏ.
  - **Nhóm 5 đặc trưng định danh (Không tính Z-score)**: Bao gồm `gender`, `region`, `disability`, `code_module`, `code_presentation`. Sau khi đi qua One-Hot Encoder, dữ liệu đã bị bẻ thành các cột nhị phân chỉ chứa giá trị `0` và `1`. Nhóm này được bỏ qua bước Z-score nhằm duy trì nguyên vẹn bản chất dữ liệu thưa (sparse) vốn đã cực kỳ đồng đều về biên độ.
## 4. Huấn luyện, đánh giá và lựa chọn mô hình (Model Selection & Tuning)

Hệ thống triển khai huấn luyện song song hai thuật toán cốt lõi thông qua kiến trúc ML Pipeline tự động:
- **Random Forest Classifier**: Thuật toán dựa trên cây quyết định, có khả năng nắm bắt tốt các mối quan hệ phi tuyến tính phức tạp.
- **LinearSVC (Support Vector Machine)**: Thuật toán tuyến tính tốc độ cao, yêu cầu bổ sung bước tiền xử lý chuẩn hóa dữ liệu (`StandardScaler`).

### 4.1. Tối ưu và Lựa chọn đặc trưng (Feature Selection & Regularization)
Khác với phương pháp tiền xử lý cũ là loại bỏ thủ công các biến có Gini Importance < 1% (như tình trạng khuyết tật `disability`), phiên bản kiến trúc mới **giữ lại toàn bộ 24 đặc trưng**. Việc kiểm soát nhiễu cục bộ và ngăn chặn triệt để hiện tượng quá khớp (Overfitting) được giao cho chính các cơ chế nội tại của thuật toán:
- **Với Random Forest**: Hệ thống sử dụng phương pháp Cost-Complexity Pruning (tỉa cây qua tham số `ccp_alpha`) để loại bỏ các nhánh cây học thuộc lòng dữ liệu nhiễu.
- **Với LinearSVC**: Tự động giảm trọng số của các biến ít đóng góp thông qua cơ chế phạt L2 Regularization (điều khiển qua siêu tham số `C`).

### 4.2. Đánh giá và Tối ưu hóa (Hyperparameter Tuning)
- Sử dụng Stratified 5-Fold Cross-Validation để so sánh hiệu năng trực diện (Head-to-Head). Kết quả thực nghiệm chỉ ra rằng **LinearSVC** đem lại sự ổn định tuyệt đối (Overfit Gap cực thấp) và tốc độ huấn luyện siêu tốc. Ngược lại, **Random Forest** nhỉnh hơn một chút về độ chính xác (Accuracy) và F1-Score do bản chất dữ liệu có nhiều mối liên hệ phi tuyến.
- Thuật toán `GridSearchCV` được sử dụng để quét tìm cấu hình thông số tối ưu nhất cho từng mô hình. Các tham số cấu hình này được lưu trữ tự động vào `best_rf_params.json` và `best_svm_params.json`. Cả hai Pipeline tốt nhất cũng được xuất ra dạng `.joblib` sẵn sàng cho môi trường Production.
## 5. Áp dụng mô hình vào hệ thống
- **Kiến trúc**: Xây dựng dưới dạng ML Pipeline trong FastAPI.
- **Quy trình**: Nhận thông tin sinh viên -> Tiền xử lý theo quy trình tại Phần 3 -> Đưa vào model đã train -> Trả về xác suất Đậu/Trượt qua API.
- **Định dạng lưu trữ**: Sử dụng `.parquet` cho dữ liệu trung gian và `.joblib` cho mô hình đã huấn luyện để tối ưu tốc độ.

## 6. Đánh giá kết quả
- Mô hình chứng minh được các yếu tố như **Tổng điểm trọng số** và **Lượt tương tác Quiz** là những chỉ báo quan trọng nhất.
- Hệ thống có khả năng dự đoán sớm ngay từ giữa kỳ học, giúp tối ưu hóa nguồn lực hỗ trợ của giáo viên.
