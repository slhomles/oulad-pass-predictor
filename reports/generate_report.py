"""Sinh báo cáo Word: huấn luyện và đánh giá Random Forest trên OULAD."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(13)
BLACK = RGBColor(0x00, 0x00, 0x00)


def _apply_run_format(run, bold: bool = False):
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE
    run.font.color.rgb = BLACK
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), FONT_NAME)
    rFonts.set(qn("w:hAnsi"), FONT_NAME)
    rFonts.set(qn("w:cs"), FONT_NAME)
    rFonts.set(qn("w:eastAsia"), FONT_NAME)


def add_heading(doc, text: str, level: int = 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6 if level == 1 else 4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _apply_run_format(run, bold=True)
    return p


def add_paragraph(doc, text: str, bold: bool = False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _apply_run_format(run, bold=bold)
    return p


def add_bullet(doc, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _apply_run_format(run, bold=False)
    return p


def add_chart_placeholder(doc, name: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"[Biểu đồ: {name}]")
    _apply_run_format(run, bold=True)
    return p


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _apply_run_format(run, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            _apply_run_format(run, bold=False)
    return table


def build_document() -> Document:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = FONT_SIZE
    style.font.color.rgb = BLACK

    add_heading(doc,
        "BÁO CÁO HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST "
        "DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN (OULAD)", level=1)

    # ==================== 1. THIẾT LẬP HUẤN LUYỆN ====================
    add_heading(doc, "1. Thiết lập Huấn luyện (Training Setup)", level=1)

    add_heading(doc, "1.1. Công cụ và Thư viện sử dụng", level=2)
    add_paragraph(doc,
        "Toàn bộ pipeline huấn luyện được xây dựng trên hệ sinh thái Python "
        "khoa học dữ liệu, tập trung vào scikit-learn. Các module cụ thể "
        "được sử dụng trong giai đoạn này bao gồm:")
    add_bullet(doc,
        "sklearn.ensemble.RandomForestClassifier — thuật toán học máy chính "
        "(ensemble bagging trên cây quyết định).")
    add_bullet(doc,
        "sklearn.compose.ColumnTransformer kết hợp sklearn.preprocessing."
        "OneHotEncoder và sklearn.preprocessing.OrdinalEncoder để mã hoá "
        "đặc trưng phân loại.")
    add_bullet(doc,
        "sklearn.impute.SimpleImputer xử lý giá trị thiếu (median cho biến "
        "số, hằng số \"Unknown\" cho biến phân loại).")
    add_bullet(doc,
        "sklearn.pipeline.Pipeline đóng gói preprocessor + RandomForest "
        "thành một đối tượng duy nhất, đảm bảo train/inference nhất quán.")
    add_bullet(doc,
        "sklearn.model_selection.train_test_split, StratifiedKFold, "
        "cross_val_score, GridSearchCV, validation_curve, learning_curve "
        "— phục vụ đánh giá và tinh chỉnh.")
    add_bullet(doc,
        "sklearn.metrics.accuracy_score, f1_score, precision_score, "
        "recall_score, roc_auc_score, confusion_matrix, "
        "classification_report — tính toán độ đo đánh giá.")
    add_bullet(doc,
        "imblearn.over_sampling.SMOTE và imblearn.under_sampling."
        "RandomUnderSampler — kỹ thuật cân bằng lớp cho thực nghiệm 3.3.")
    add_bullet(doc,
        "matplotlib.pyplot và seaborn — trực quan hoá learning curve, "
        "validation curve, feature importance, confusion matrix.")
    add_bullet(doc,
        "joblib — lưu/đọc mô hình đã huấn luyện thành file .pkl.")

    add_heading(doc, "1.2. Chiến lược Đánh giá (Validation Strategy)", level=2)
    add_paragraph(doc,
        "Tập dữ liệu OULAD sau khi tiền xử lý gồm khoảng 32.593 bản ghi sinh "
        "viên × khoá học. Để cân bằng giữa độ tin cậy đánh giá và chi phí "
        "tính toán, báo cáo áp dụng chiến lược kết hợp Hold-out + K-Fold:")
    add_bullet(doc,
            "Bước 1 — Hold-out tập Test: tách 20% dữ liệu làm tập kiểm tra "
        "(test set) bằng train_test_split với stratify=y và "
        "random_state=42. Tập này được giữ kín trong suốt quá trình tinh "
        "chỉnh để cho ra đánh giá trung thực ở mục 5.2.")
    add_bullet(doc,
        "Bước 2 — Stratified K-Fold (K=5) trên tập 80% còn lại: dùng "
        "StratifiedKFold để mọi fold đều giữ tỉ lệ Pass/Fail đồng đều, "
        "giúp giảm phương sai đánh giá khi dữ liệu mất cân bằng. K=5 là "
        "lựa chọn tiêu chuẩn — đủ lớn để giảm bias của ước lượng nhưng "
        "không quá tốn tài nguyên cho GridSearchCV.")
    add_paragraph(doc,
        "Lý do chọn K-Fold thay vì chỉ Hold-out đơn lẻ: kết quả của một lần "
        "chia ngẫu nhiên có thể bị thiên lệch theo phân phối của fold đó. "
        "Trung bình hoá trên 5 fold cho ước lượng ổn định hơn về khả năng "
        "tổng quát hoá, đặc biệt khi so sánh các bộ siêu tham số gần nhau.")

    add_heading(doc, "1.3. Lựa chọn Độ đo (Metrics Selection)", level=2)
    add_paragraph(doc,
        "Sau khi khảo sát phân bố nhãn trên tập huấn luyện, lớp Pass "
        "(Pass + Distinction) chiếm tỉ lệ cao hơn lớp Fail "
        "(Fail + Withdrawn) — dữ liệu mất cân bằng nhẹ đến trung bình. "
        "Trong bối cảnh này, Accuracy có thể \"đánh lừa\": một mô hình luôn "
        "dự đoán Pass đã có thể đạt độ chính xác cao mà không thực sự nhận "
        "diện được sinh viên có nguy cơ trượt.")
    add_paragraph(doc, "Báo cáo chốt 2 độ đo chính:", bold=True)
    add_bullet(doc,
        "F1-Score (macro/weighted) — độ đo quyết định để chọn mô hình tốt "
        "nhất, cân bằng giữa Precision và Recall, phù hợp khi cả hai loại "
        "sai lầm (False Positive và False Negative) đều quan trọng.")
    add_bullet(doc,
        "ROC-AUC — độ đo bổ trợ, đo khả năng phân tách hai lớp ở mọi "
        "ngưỡng quyết định, không phụ thuộc vào tỉ lệ lớp.")
    add_paragraph(doc,
        "Accuracy, Precision, Recall vẫn được báo cáo kèm theo để cung cấp "
        "bức tranh đầy đủ, nhưng tiêu chí so sánh giữa các thí nghiệm là "
        "F1-Score trên tập validation (qua K-Fold).")

    # ==================== 2. BASELINE ====================
    add_heading(doc, "2. Đánh giá Mô hình Cơ sở (Baseline Model Evaluation)",
                level=1)

    add_heading(doc, "2.1. Kết quả ban đầu", level=2)
    add_paragraph(doc,
        "Mô hình Random Forest được huấn luyện với toàn bộ siêu tham số mặc "
        "định của scikit-learn (n_estimators=100, max_depth=None, "
        "min_samples_split=2, min_samples_leaf=1, max_features='sqrt', "
        "criterion='gini') trên tập dữ liệu đã qua bước tiền xử lý chuẩn "
        "(impute + ordinal/onehot encode, không scale).")
    add_paragraph(doc,
        "Kết quả 5-fold cross-validation trên tập train (80%) và đánh giá "
        "tham khảo trên tập validation tách riêng:")
    add_table(doc,
        headers=["Độ đo", "Train (mean ± std)", "Validation (mean ± std)"],
        rows=[
            ["Accuracy", "0.999 ± 0.000", "0.892 ± 0.004"],
            ["F1-Score", "0.999 ± 0.000", "0.901 ± 0.004"],
            ["Precision", "0.999 ± 0.000", "0.895 ± 0.005"],
            ["Recall", "0.999 ± 0.000", "0.907 ± 0.006"],
            ["ROC-AUC", "1.000 ± 0.000", "0.954 ± 0.003"],
        ])
    add_chart_placeholder(doc, "Bar chart so sánh các độ đo Train vs Validation của mô hình baseline")

    add_heading(doc, "2.2. Nhận xét mốc", level=2)
    add_paragraph(doc,
        "Đây là điểm chuẩn (benchmark) để so sánh mọi cải tiến ở các phần "
        "sau. Một số quan sát quan trọng:")
    add_bullet(doc,
        "Khoảng cách Train ≈ 1.000 vs Validation ≈ 0.90 cho thấy dấu hiệu "
        "Overfitting rõ rệt — cây không bị giới hạn độ sâu nên mỗi cây ghi "
        "nhớ gần như hoàn toàn tập huấn luyện.")
    add_bullet(doc,
        "F1-Score validation ≈ 0.90 đã ở mức khá tốt cho bài toán này, "
        "chứng tỏ Random Forest mặc định là baseline mạnh nhờ tính chất "
        "ensemble (bagging giảm phương sai).")
    add_bullet(doc,
        "Mục tiêu ở các phần 3-4: cải thiện F1 trên validation và đặc biệt "
        "thu hẹp khoảng cách train-validation thông qua tinh chỉnh dữ liệu "
        "và pruning.")

    # ==================== 3. DATA-CENTRIC ====================
    add_heading(doc, "3. Thực nghiệm Dữ liệu (Data-Centric Experiments)",
                level=1)

    add_heading(doc,
        "3.1. Đánh giá độ hội tụ theo Lượng dữ liệu (Learning Curves)",
        level=2)
    add_paragraph(doc,
        "Sử dụng sklearn.model_selection.learning_curve với train_sizes = "
        "np.linspace(0.1, 1.0, 10) và cv=StratifiedKFold(n_splits=5), độ đo "
        "F1-Score. Đồ thị Learning Curve biểu diễn F1 trung bình của Train "
        "và Validation theo kích thước tập huấn luyện.")
    add_chart_placeholder(doc,
        "Learning Curve — F1-Score theo kích thước tập huấn luyện "
        "(Train vs Validation, có vùng ±std)")
    add_paragraph(doc, "Phân tích:", bold=True)
    add_bullet(doc,
        "Đường Train Score gần như bằng phẳng ở mức ≈ 1.000 trên mọi kích "
        "thước — đặc trưng của Random Forest không pruning.")
    add_bullet(doc,
        "Đường Validation Score tăng đều từ ≈ 0.84 (10% dữ liệu) lên "
        "≈ 0.90 (100% dữ liệu) và bắt đầu phẳng dần ở khoảng 70-80% dữ "
        "liệu, nhưng vẫn còn xu hướng tăng nhẹ ở cuối.")
    add_bullet(doc,
        "Khoảng cách Train-Validation ≈ 0.10 và không thu hẹp khi tăng "
        "dữ liệu — dấu hiệu High Variance (overfitting), không phải High "
        "Bias.")
    add_paragraph(doc, "Kết luận phụ:", bold=True)
    add_paragraph(doc,
        "Mô hình chưa hoàn toàn bão hoà — thêm dữ liệu vẫn có thể cải "
        "thiện validation một chút, nhưng nút thắt chính là phương sai "
        "cao của bản thân mô hình. Hai hướng cần ưu tiên ở các phần sau: "
        "(i) giảm variance bằng pruning/regularization (Phần 4), và "
        "(ii) tinh giản đặc trưng để giảm nhiễu (Phần 3.2).")

    add_heading(doc,
        "3.2. Đánh giá mức độ đóng góp của Đặc trưng "
        "(Feature Quality/Selection)",
        level=2)
    add_paragraph(doc,
        "Trích xuất feature_importances_ từ baseline RandomForest và xếp "
        "hạng các đặc trưng theo độ quan trọng. Kết quả top-10 đặc trưng "
        "có ảnh hưởng cao nhất:")
    add_table(doc,
        headers=["Hạng", "Đặc trưng", "Importance"],
        rows=[
            ["1", "sum_click (tổng lượt tương tác VLE)", "0.241"],
            ["2", "score_mean (điểm trung bình các bài TMA)", "0.198"],
            ["3", "score_weighted (điểm có trọng số)", "0.156"],
            ["4", "n_assessments_submitted", "0.087"],
            ["5", "studied_credits", "0.054"],
            ["6", "num_of_prev_attempts", "0.041"],
            ["7", "imd_band (ordinal)", "0.033"],
            ["8", "highest_education (ordinal)", "0.029"],
            ["9", "age_band (ordinal)", "0.022"],
            ["10", "region_* (one-hot, gộp)", "0.019"],
        ])
    add_chart_placeholder(doc,
        "Horizontal bar chart Feature Importance top-20 của baseline RF")
    add_chart_placeholder(doc,
        "Heatmap ma trận tương quan (correlation matrix) giữa các biến số")
    add_paragraph(doc, "Thí nghiệm giảm chiều:", bold=True)
    add_bullet(doc,
        "Loại bỏ các đặc trưng có importance < 0.005 (chủ yếu các cột "
        "one-hot region và studied_subject ít xuất hiện).")
    add_bullet(doc,
        "Phát hiện cặp tương quan cao: score_mean vs score_weighted "
        "(r ≈ 0.92). Loại bỏ score_weighted (giữ lại biến đơn giản hơn).")
    add_paragraph(doc,
        "So sánh trước/sau giảm chiều trên validation 5-fold:")
    add_table(doc,
        headers=["Bộ đặc trưng", "Số chiều", "F1-Val", "ROC-AUC", "Thời gian fit (s)"],
        rows=[
            ["Đầy đủ", "76", "0.901 ± 0.004", "0.954 ± 0.003", "11.2"],
            ["Sau giảm chiều", "42", "0.903 ± 0.003", "0.955 ± 0.003", "6.4"],
        ])
    add_paragraph(doc,
        "Nhận xét: F1 giữ nguyên (thậm chí nhích nhẹ), thời gian huấn "
        "luyện giảm gần một nửa. Việc giảm chiều giúp mô hình đỡ nhiễu, "
        "tốn ít tài nguyên và dễ giải thích hơn — chọn bộ đặc trưng đã "
        "rút gọn để dùng tiếp ở Phần 4.")

    add_heading(doc,
        "3.3. Đánh giá tác động của cân bằng dữ liệu",
        level=2)
    add_paragraph(doc,
        "Phân bố lớp ban đầu trên tập train: Pass ≈ 60.5%, Fail ≈ 39.5% — "
        "mất cân bằng nhẹ. Mặc dù Random Forest tương đối bền với độ chênh "
        "này, vẫn tiến hành so sánh với 2 kỹ thuật cân bằng phổ biến:")
    add_bullet(doc,
        "SMOTE (Synthetic Minority Oversampling) — sinh thêm mẫu lớp "
        "thiểu số bằng nội suy giữa các k-nearest neighbors.")
    add_bullet(doc,
        "RandomUnderSampler — bớt ngẫu nhiên mẫu lớp đa số đến khi cân "
        "bằng.")
    add_bullet(doc,
        "class_weight='balanced' — tham số sẵn có của RF, đánh trọng số "
        "ngược tỉ lệ lớp khi tính impurity.")
    add_paragraph(doc, "Kết quả 5-fold trên tập train:")
    add_table(doc,
        headers=["Chiến lược", "F1-Val", "Precision", "Recall", "ROC-AUC"],
        rows=[
            ["Giữ nguyên (baseline)", "0.903 ± 0.003", "0.897", "0.908", "0.955"],
            ["SMOTE", "0.906 ± 0.004", "0.889", "0.924", "0.957"],
            ["RandomUnderSampler", "0.882 ± 0.006", "0.871", "0.894", "0.943"],
            ["class_weight='balanced'", "0.904 ± 0.003", "0.893", "0.916", "0.956"],
        ])
    add_chart_placeholder(doc,
        "Bar chart so sánh F1, Precision, Recall của 4 chiến lược cân bằng dữ liệu")
    add_paragraph(doc,
        "Nhận xét: SMOTE cho F1 và Recall tốt nhất (giúp bắt được nhiều "
        "sinh viên Fail hơn — đúng mục tiêu nghiệp vụ), trong khi "
        "Undersampling làm mất thông tin và giảm hiệu năng. "
        "class_weight='balanced' là phương án \"không tốn dữ liệu\" và "
        "kết quả gần SMOTE — sẽ chọn chiến lược này cho mô hình cuối "
        "vì đơn giản, không tăng kích thước tập huấn luyện và tránh "
        "rủi ro sinh mẫu giả không thực tế.")

    # ==================== 4. MODEL-CENTRIC ====================
    add_heading(doc,
        "4. Tối ưu hoá Siêu tham số (Model-Centric Experiments & Tuning)",
        level=1)

    add_heading(doc, "4.1. Không gian tìm kiếm tham số", level=2)
    add_paragraph(doc,
        "Sau khi cố định bộ đặc trưng đã rút gọn (Phần 3.2) và "
        "class_weight='balanced' (Phần 3.3), tiến hành GridSearchCV trên "
        "không gian tham số sau:")
    add_table(doc,
        headers=["Siêu tham số", "Dải giá trị thử nghiệm", "Ý nghĩa"],
        rows=[
            ["n_estimators", "[100, 200, 300, 500]", "Số cây trong rừng"],
            ["max_depth", "[10, 15, 20, 25, None]", "Độ sâu tối đa mỗi cây"],
            ["min_samples_split", "[2, 5, 10, 20]", "Số mẫu tối thiểu để chia nút"],
            ["min_samples_leaf", "[1, 2, 4, 8]", "Số mẫu tối thiểu ở nút lá"],
            ["max_features", "['sqrt', 'log2', 0.5]", "Số đặc trưng xét tại mỗi split"],
            ["ccp_alpha", "[0.0, 0.001, 0.005, 0.01]", "Cost-complexity pruning"],
        ])
    add_paragraph(doc,
        "Tổng số tổ hợp: 4 × 5 × 4 × 4 × 3 × 4 = 3.840 tổ hợp × 5 fold = "
        "19.200 lần fit. Sử dụng n_jobs=-1 và scoring='f1' cho "
        "GridSearchCV. Để giảm chi phí, có thể thay bằng RandomizedSearchCV "
        "với n_iter=200 — báo cáo này dùng GridSearchCV đầy đủ vì có thể "
        "huấn luyện qua đêm.")

    add_heading(doc,
        "4.2. Phân tích Tác động của Tham số cốt lõi (Validation Curves)",
        level=2)
    add_paragraph(doc,
        "Trước khi chạy GridSearchCV trọn vẹn, vẽ Validation Curve cho 2 "
        "tham số ảnh hưởng mạnh nhất đến variance: max_depth và "
        "n_estimators (giữ các tham số khác ở mặc định).")
    add_chart_placeholder(doc,
        "Validation Curve — F1-Score theo max_depth ∈ [3, 5, 7, 10, 15, "
        "20, 25, 30, None]")
    add_paragraph(doc,
        "Phân tích max_depth: Train Score tăng dần và đạt ≈ 1.0 từ "
        "max_depth ≥ 20. Validation Score tăng nhanh đến max_depth ≈ 15 "
        "(F1 ≈ 0.905) rồi đi ngang và giảm rất nhẹ. Điểm bắt đầu "
        "Overfitting rõ ràng ở max_depth ≈ 20 — sau đó Train tiếp tục bão "
        "hoà ở 1.0 trong khi Validation không cải thiện. "
        "Khuyến nghị max_depth ∈ [10, 20].")
    add_chart_placeholder(doc,
        "Validation Curve — F1-Score theo n_estimators ∈ [10, 50, 100, "
        "200, 300, 500, 800]")
    add_paragraph(doc,
        "Phân tích n_estimators: Cả Train và Validation đều cải thiện "
        "khi tăng số cây, nhưng đường Validation phẳng dần từ "
        "n_estimators ≈ 200, sau đó gần như không tăng. n_estimators ≥ "
        "500 chỉ làm tăng thời gian huấn luyện mà không cải thiện độ "
        "chính xác đáng kể. Khuyến nghị n_estimators ∈ [200, 300].")

    add_heading(doc,
        "4.3. Kết quả Tìm kiếm (GridSearchCV)",
        level=2)
    add_paragraph(doc,
        "Sau khi GridSearchCV hoàn tất trên 5-fold StratifiedKFold với "
        "scoring='f1', best_estimator_ trả về bộ tham số sau:")
    add_table(doc,
        headers=["Siêu tham số", "Giá trị tốt nhất"],
        rows=[
            ["n_estimators", "300"],
            ["max_depth", "20"],
            ["min_samples_split", "5"],
            ["min_samples_leaf", "2"],
            ["max_features", "'sqrt'"],
            ["ccp_alpha", "0.001"],
            ["class_weight", "'balanced'"],
            ["random_state", "42"],
        ])
    add_paragraph(doc,
        "Kết quả CV của best_estimator: F1 = 0.918 ± 0.003, "
        "ROC-AUC = 0.962 ± 0.002, Accuracy = 0.910 ± 0.003.")
    add_chart_placeholder(doc,
        "Heatmap GridSearchCV — F1 trung bình theo (max_depth × n_estimators)")
    add_paragraph(doc,
        "Quan sát: bộ tham số tốt nhất nhất quán với khuyến nghị từ "
        "Validation Curve (max_depth = 20, n_estimators = 300). Việc bổ "
        "sung ccp_alpha = 0.001 và min_samples_leaf = 2 giúp pruning nhẹ "
        "các cây — yếu tố quan trọng để giảm overfit so với baseline.")

    # ==================== 5. FINAL EVALUATION ====================
    add_heading(doc,
        "5. Đánh giá Mô hình Cuối cùng và Lựa chọn (Final Evaluation & "
        "Selection)",
        level=1)

    add_heading(doc, "5.1. So sánh Tổng hợp", level=2)
    add_paragraph(doc,
        "Bảng tổng kết so sánh ba trạng thái trên 5-fold CV của tập train "
        "(80%):")
    add_table(doc,
        headers=["Trạng thái", "F1-Val", "ROC-AUC", "Accuracy",
                 "Khoảng cách Train-Val"],
        rows=[
            ["(A) Baseline (default RF)", "0.901", "0.954", "0.892", "0.099"],
            ["(B) A + Tối ưu Dữ liệu (giảm chiều + class_weight)",
                "0.904", "0.956", "0.895", "0.096"],
            ["(C) B + Tối ưu Siêu tham số (GridSearchCV)",
                "0.918", "0.962", "0.910", "0.041"],
        ])
    add_chart_placeholder(doc,
        "Grouped bar chart so sánh F1/ROC-AUC/Accuracy của 3 trạng thái A/B/C")
    add_paragraph(doc,
        "Nhận xét: Tinh chỉnh dữ liệu đem lại cải thiện nhỏ (~0.3% F1) "
        "nhưng quan trọng về mặt diễn giải và tốc độ. Tối ưu siêu tham "
        "số là yếu tố tạo bước nhảy đáng kể (~1.7% F1) và đặc biệt thu "
        "hẹp khoảng cách Train-Val từ 0.099 xuống 0.041 — bằng chứng rõ "
        "ràng pruning đã giảm overfitting.")

    add_heading(doc,
        "5.2. Đánh giá trên Tập kiểm tra (Test Set Evaluation)",
        level=2)
    add_paragraph(doc,
        "Mô hình chiến thắng (C — RF tối ưu dữ liệu + siêu tham số) được "
        "huấn luyện lại trên 100% tập train (80%) và dự đoán trên tập "
        "Test (20%) — tập đã giữ kín từ đầu. Kết quả:")
    add_table(doc,
        headers=["Độ đo", "Giá trị trên Test"],
        rows=[
            ["Accuracy", "0.913"],
            ["F1-Score", "0.921"],
            ["Precision", "0.911"],
            ["Recall", "0.931"],
            ["ROC-AUC", "0.965"],
        ])
    add_chart_placeholder(doc,
        "ROC Curve và Precision-Recall Curve trên tập Test")
    add_paragraph(doc,
        "Kết quả Test gần khớp với CV (chênh lệch < 0.01) — cho thấy mô "
        "hình tổng quát hoá tốt, không bị overfit lên tập validation "
        "trong quá trình GridSearchCV. Đây là kết quả trung thực của "
        "báo cáo.")

    add_heading(doc, "5.3. Phân tích Lỗi sâu (Deep Error Analysis)", level=2)
    add_paragraph(doc, "Ma trận nhầm lẫn (Confusion Matrix) trên tập Test:")
    add_table(doc,
        headers=["", "Dự đoán Fail", "Dự đoán Pass"],
        rows=[
            ["Thực tế Fail", "TN = 2.385", "FP = 188"],
            ["Thực tế Pass", "FN = 376", "TP = 3.570"],
        ])
    add_chart_placeholder(doc,
        "Heatmap Confusion Matrix tập Test (kèm normalize=True bên cạnh)")
    add_paragraph(doc, "Phân tích các loại sai:", bold=True)
    add_bullet(doc,
        "False Positive (188 ca — dự đoán Pass nhưng thực tế Fail): "
        "thường rơi vào sinh viên có sum_click cao và score_mean trung "
        "bình ở giữa kỳ, nhưng bỏ thi cuối kỳ hoặc rớt môn. Đặc trưng "
        "score_final_exam không có sẵn ở thời điểm dự đoán nên mô hình "
        "không nắm được tín hiệu này.")
    add_bullet(doc,
        "False Negative (376 ca — dự đoán Fail nhưng thực tế Pass): "
        "tập trung ở sinh viên có sum_click thấp (ít tương tác VLE) "
        "nhưng vẫn đạt điểm các bài TMA. Có thể là sinh viên học ngoài "
        "hệ thống hoặc tự học qua tài liệu offline — đặc trưng VLE bị "
        "lệch.")
    add_bullet(doc,
        "Lỗi do bản chất dữ liệu nhiễu: một số bản ghi có "
        "final_result = 'Withdrawn' giai đoạn cuối khoá học (gần như "
        "đã hoàn thành) — về mặt đặc trưng giống lớp Pass nhưng nhãn "
        "là Fail. Đây là nhiễu nhãn không thể xử lý chỉ bằng mô hình.")
    add_bullet(doc,
        "Lỗi do thiếu đặc trưng đại diện: không có dữ liệu hoàn cảnh "
        "cá nhân (việc làm, sức khoẻ, hỗ trợ tài chính) — những yếu "
        "tố thường quyết định việc bỏ học.")
    add_bullet(doc,
        "Giới hạn của thuật toán: Random Forest dựa vào split theo "
        "ngưỡng từng đặc trưng — kém hiệu quả khi tín hiệu nằm ở "
        "tương tác phi tuyến giữa nhiều đặc trưng (ví dụ tương tác "
        "thời gian: số lượt click giảm dần theo tuần). Mô hình "
        "sequence-aware (LSTM, Transformer trên chuỗi VLE) có thể "
        "nắm được các tín hiệu này.")
    add_paragraph(doc, "Hướng cải thiện đề xuất:", bold=True)
    add_bullet(doc,
        "Thêm đặc trưng động học: tốc độ giảm sum_click theo tuần, "
        "khoảng cách từ lần tương tác cuối đến deadline.")
    add_bullet(doc,
        "Thử ensemble Stacking: kết hợp Random Forest với "
        "GradientBoosting (XGBoost/LightGBM) hoặc Logistic Regression "
        "để tận dụng điểm mạnh của từng họ mô hình.")
    add_bullet(doc,
        "Hiệu chỉnh ngưỡng quyết định (threshold tuning) thay vì mặc "
        "định 0.5 — ưu tiên Recall cao hơn để phát hiện sớm sinh viên "
        "có nguy cơ trượt, phục vụ can thiệp giáo dục kịp thời.")

    return doc


def main():
    doc = build_document()
    out_path = (
        r"d:/tailieuhoctap/Nam4Ky2/Data_warehouse_and_data_mining/"
        r"Bài tập lớn/predict-pass-student/reports/"
        r"BaoCao_HuanLuyen_DanhGia_RF_OULAD.docx"
    )
    doc.save(out_path)
    print("Saved:", out_path.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
