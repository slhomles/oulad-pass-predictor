"""Sinh file Word báo cáo phần Huấn luyện & Đánh giá RF vs SVM (OULAD).

Sử dụng kết quả thực từ notebooks/03_rf_svm_comparison.ipynb.
Format: Times New Roman 13pt — heading bold, body normal.
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(13)
BLACK = RGBColor(0x00, 0x00, 0x00)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _set_run_font(run, bold: bool = False):
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE
    run.font.color.rgb = BLACK
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), FONT_NAME)


def add_heading(doc, text: str, level: int = 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8 if level == 1 else 5)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _set_run_font(run, bold=True)
    return p


def add_body(doc, text: str, bold: bool = False, indent: float = 0):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    run = p.add_run(text)
    _set_run_font(run, bold=bold)
    return p


def add_bullet(doc, text: str, level: int = 1):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Pt(18 * level)
    run = p.add_run(text)
    _set_run_font(run, bold=False)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    # header row
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _set_run_font(run, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # data rows
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            _set_run_font(run, bold=False)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    return table


def add_figure(doc, filename: str, caption: str, width: float = 5.5):
    path = FIG / filename
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width))
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"[Biểu đồ: {filename}]")
        _set_run_font(run, bold=True)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(6)
    run2 = cap.add_run(caption)
    _set_run_font(run2, bold=False)
    run2.font.size = Pt(12)
    run2.font.italic = True
    return p


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

def build_document() -> Document:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = FONT_SIZE
    style.font.color.rgb = BLACK

    # Tiêu đề tổng
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(8)
    run_title = p_title.add_run(
        "PHẦN III. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH\n"
        "SO SÁNH RANDOM FOREST VÀ SUPPORT VECTOR MACHINE TRÊN BỘ DỮ LIỆU OULAD"
    )
    _set_run_font(run_title, bold=True)

    # =========================================================================
    # 1. THIẾT LẬP HUẤN LUYỆN
    # =========================================================================
    add_heading(doc, "1. Thiết lập Huấn luyện", level=1)

    add_heading(doc, "1.1. Dữ liệu và Phân chia tập Train/Test", level=2)
    add_body(doc,
        "Bộ dữ liệu OULAD sau khi tiền xử lý ở Phần II gồm 32.593 bản ghi "
        "sinh viên – khoá học, được biểu diễn bằng 24 đặc trưng thô (raw features) "
        "bao gồm 16 biến số (numeric), 3 biến thứ tự (ordinal) và 5 biến danh nghĩa "
        "(nominal). Nhãn dự đoán là biến nhị phân: 1 (Pass / Distinction) chiếm "
        "47,2% và 0 (Fail / Withdrawn) chiếm 52,8%."
    )
    add_body(doc,
        "Tập dữ liệu được chia thành:"
    )
    add_bullet(doc,
        "Tập huấn luyện (train): 80% = 26.074 mẫu — dùng để huấn luyện "
        "và xác thực chéo trong quá trình tinh chỉnh siêu tham số."
    )
    add_bullet(doc,
        "Tập kiểm tra (test): 20% = 6.519 mẫu — giữ kín hoàn toàn đến "
        "Mục 5.3 để đánh giá trung thực hiệu năng cuối cùng."
    )
    add_body(doc,
        "Phân chia được thực hiện bằng train_test_split với stratify=y "
        "và random_state=42, đảm bảo tỉ lệ Pass/Fail đồng đều ở cả hai tập "
        "(tập train: 47,2%, tập test: 47,2%)."
    )

    add_figure(doc, "03_target_distribution.png",
               "Hình 1: Phân phối nhãn (Pass/Fail) và đặc trưng số trên tập train")

    add_heading(doc, "1.2. Chiến lược Xác thực Chéo", level=2)
    add_body(doc,
        "Mọi thí nghiệm huấn luyện trong báo cáo này đều sử dụng Stratified "
        "K-Fold với K=5 (StratifiedKFold, shuffle=True, random_state=42). "
        "Chiến lược này đảm bảo:"
    )
    add_bullet(doc,
        "Mỗi fold duy trì tỉ lệ lớp Pass/Fail xấp xỉ bằng nhau (~47,2%/52,8%), "
        "tránh thiên lệch đánh giá khi dữ liệu có mất cân bằng nhẹ."
    )
    add_bullet(doc,
        "Trung bình 5 fold cung cấp ước lượng khả năng tổng quát hoá ổn định "
        "hơn so với một lần chia Hold-out đơn lẻ."
    )
    add_bullet(doc,
        "Độ lệch chuẩn giữa các fold (±std) phản ánh mức độ ổn định "
        "(stability) của mô hình — tiêu chí phụ để phân biệt khi hai mô hình "
        "có F1 gần nhau."
    )

    add_heading(doc, "1.3. Độ đo Đánh giá", level=2)
    add_body(doc,
        "Báo cáo sử dụng bộ 5 độ đo đánh giá cho mỗi thí nghiệm:"
    )
    add_table(doc,
        headers=["Độ đo", "Ký hiệu", "Vai trò trong báo cáo"],
        rows=[
            ["F1-Score", "F1", "Độ đo quyết định — cân bằng Precision/Recall"],
            ["ROC-AUC", "AUC", "Khả năng phân tách 2 lớp ở mọi ngưỡng"],
            ["Accuracy", "Acc", "Tỉ lệ dự đoán đúng tổng thể"],
            ["Precision", "P", "Tỉ lệ dự đoán Pass đúng trong số dự đoán Pass"],
            ["Recall", "R", "Tỉ lệ phát hiện đúng sinh viên thực tế Pass"],
        ],
        col_widths=[1.5, 0.8, 3.8]
    )
    add_body(doc,
        "F1-Score là tiêu chí chính để so sánh và chọn mô hình vì nó cân bằng "
        "giữa Precision (tránh báo động giả) và Recall (phát hiện đủ sinh viên "
        "có nguy cơ trượt) — cả hai đều quan trọng trong bài toán hỗ trợ can "
        "thiệp giáo dục kịp thời."
    )

    # =========================================================================
    # 2. PHÂN NHÁNH PIPELINE: RF vs SVM
    # =========================================================================
    add_heading(doc, "2. Kiến trúc Pipeline: Random Forest và SVM", level=1)

    add_heading(doc, "2.1. Tiền xử lý chung (Preprocessor)", level=2)
    add_body(doc,
        "Cả hai thuật toán đều dùng chung một ColumnTransformer xử lý 24 đặc "
        "trưng thô theo 3 nhánh:"
    )
    add_bullet(doc,
        "Biến số (16 cột): SimpleImputer(strategy='median') — điền giá trị "
        "thiếu bằng trung vị, không thay đổi phân phối."
    )
    add_bullet(doc,
        "Biến thứ tự ordinal (3 cột: highest_education, age_band, imd_band): "
        "SimpleImputer(fill_value='Unknown') → OrdinalEncoder với thứ tự "
        "giáo dục định nghĩa sẵn."
    )
    add_bullet(doc,
        "Biến danh nghĩa nominal (5 cột: gender, region, disability, "
        "code_module, code_presentation): SimpleImputer → OneHotEncoder → "
        "28 cột nhị phân."
    )
    add_body(doc,
        "Sau khi qua preprocessor, mỗi mẫu được biểu diễn bằng 47 đặc trưng "
        "(16 numeric + 3 ordinal + 28 OHE)."
    )

    add_heading(doc, "2.2. Phân nhánh Pipeline theo Thuật toán", level=2)
    add_body(doc,
        "Hai thuật toán yêu cầu hai kiến trúc pipeline khác nhau do tính chất "
        "toán học:"
    )
    add_table(doc,
        headers=["Thuộc tính", "Random Forest", "SVM (RBF / Linear)"],
        rows=[
            ["Nền tảng toán học", "Ensemble cây quyết định (chia theo ngưỡng)", "Tối ưu hoá biên (khoảng cách đến hyperplane)"],
            ["Cần chuẩn hoá scale?", "Không — RF bất biến với scale", "Có — khoảng cách phụ thuộc magnitude"],
            ["StandardScaler trong Pipeline", "Không có", "Có (đặt sau preprocessor, trước clf)"],
            ["Pipeline", "prep → RF", "prep → StandardScaler → SVC"],
            ["predict_proba", "Sẵn có", "SVC(probability=True) — dùng Platt scaling"],
        ],
        col_widths=[2.0, 2.2, 2.2]
    )
    add_body(doc,
        "Việc đặt StandardScaler bên trong sklearn.Pipeline là bắt buộc để "
        "tránh rò rỉ dữ liệu (data leakage): scaler chỉ được fit trên phần "
        "train của mỗi fold CV, không nhìn thấy dữ liệu validation — đây là "
        "điểm khác biệt quan trọng so với cách scale trước rồi mới chia fold."
    )

    add_figure(doc, "03_scaling_before_after.png",
               "Hình 2: Phân phối đặc trưng số trước và sau StandardScaler "
               "(ví dụ 4 đặc trưng có phân phối lệch nhất)")

    add_heading(doc, "2.3. Kiểm thử Nhanh (Smoke Test)", level=2)
    add_body(doc,
        "Cả hai pipeline được kiểm thử trên 500 mẫu đầu tiên để xác nhận "
        "chạy đúng trước khi bắt đầu các thí nghiệm chính. Kết quả: "
        "RF pipeline cho shape đầu ra sau preprocessor là (500, 47); "
        "SVM pipeline dự đoán được 2 lớp {0, 1} — xác nhận pipeline hợp lệ."
    )

    # =========================================================================
    # 3. ĐÁNH GIÁ MÔ HÌNH CƠ SỞ (BASELINE)
    # =========================================================================
    add_heading(doc, "3. Đánh giá Mô hình Cơ sở (Baseline)", level=1)

    add_heading(doc, "3.1. Cấu hình Baseline", level=2)
    add_body(doc,
        "Mỗi thuật toán được chạy với cấu hình tối thiểu — gần nhất với "
        "mặc định của scikit-learn — để thiết lập mốc tham chiếu (benchmark) "
        "trước khi tinh chỉnh:"
    )
    add_bullet(doc,
        "RF baseline: RandomForestClassifier(n_estimators=100, max_depth=None, "
        "class_weight='balanced', random_state=42, n_jobs=-1)."
    )
    add_bullet(doc,
        "SVM baseline: SVC(kernel='rbf', C=1.0, gamma='scale', "
        "class_weight='balanced', probability=True, random_state=42)."
    )
    add_body(doc,
        "class_weight='balanced' được áp dụng cho cả hai để đảm bảo so sánh "
        "công bằng; với dữ liệu pass rate ≈ 47,2% (gần cân bằng) tham số "
        "này có tác động nhỏ nhưng nhất quán."
    )

    add_heading(doc, "3.2. Kết quả 5-Fold Cross-Validation", level=2)
    add_table(doc,
        headers=["Mô hình", "F1 ± std", "AUC ± std", "Accuracy ± std",
                 "Overfit gap", "Thời gian fit (s/fold)"],
        rows=[
            ["RF baseline",
             "0,9434 ± 0,0023", "0,9846 ± 0,0009", "0,9449 ± 0,0022",
             "+0,0551", "3,46"],
            ["SVM baseline (RBF, C=1)",
             "0,9390 ± 0,0032", "0,9820 ± 0,0016", "0,9397 ± 0,0033",
             "+0,0129", "168,29"],
        ],
        col_widths=[1.8, 1.5, 1.5, 1.5, 1.0, 1.5]
    )
    add_body(doc,
        "Nhận xét:"
    )
    add_bullet(doc,
        "RF baseline đạt F1=0,9434 — cao hơn SVM 0,44 điểm phần trăm (pp). "
        "Cả hai mô hình đã cho kết quả rất tốt ngay từ cấu hình mặc định, "
        "phản ánh chất lượng dữ liệu OULAD sau tiền xử lý."
    )
    add_bullet(doc,
        "RF có overfit gap lớn hơn (+0,0551 vs +0,0129): RF train accuracy "
        "đạt 1,000 (ghi nhớ hoàn toàn tập train), trong khi SVM train accuracy "
        "chỉ là 0,953 — SVM có tính regularisation nội tại cao hơn nhờ margin "
        "maximization."
    )
    add_bullet(doc,
        "Chi phí tính toán chênh lệch lớn: SVM mất 168,29 s/fold — gấp "
        "~48 lần RF (3,46 s/fold) — do độ phức tạp O(n²) đến O(n³) của "
        "kernel SVM trên 26.000 mẫu."
    )

    add_figure(doc, "03_baseline_compare.png",
               "Hình 3: So sánh F1, AUC, Accuracy và thời gian fit của "
               "RF baseline và SVM baseline (5-fold CV, trung bình ± std)")

    add_figure(doc, "03_baseline_per_fold.png",
               "Hình 4: Phân phối F1 theo từng fold và khoảng cách "
               "Train–CV cho RF và SVM baseline")

    add_body(doc,
        "Biểu đồ phân phối theo fold (Hình 4) cho thấy SVM có biến động "
        "F1 cao hơn giữa các fold (std=0,0032 so với RF std=0,0023), tuy nhiên "
        "khoảng cách Train–CV nhỏ hơn nhiều — phù hợp với overfit gap đã phân "
        "tích ở trên."
    )

    # =========================================================================
    # 4. TINH CHỈNH SIÊU THAM SỐ
    # =========================================================================
    add_heading(doc, "4. Tinh chỉnh Siêu tham số (Hyperparameter Tuning)", level=1)

    add_heading(doc, "4.1. Tinh chỉnh Random Forest", level=2)
    add_body(doc,
        "Sử dụng RandomizedSearchCV với n_iter=20, cv=5 (StratifiedKFold), "
        "scoring='f1', n_jobs=-1 trên toàn bộ 26.074 mẫu train. Không gian "
        "tìm kiếm gồm:"
    )
    add_table(doc,
        headers=["Siêu tham số", "Dải giá trị", "Ý nghĩa"],
        rows=[
            ["n_estimators", "[100, 200, 300]", "Số cây trong rừng"],
            ["max_depth", "[10, 15, 20, None]", "Độ sâu tối đa mỗi cây (None = không giới hạn)"],
            ["min_samples_split", "[2, 5, 10]", "Số mẫu tối thiểu để chia nút"],
            ["ccp_alpha", "[0,0; 0,0001; 0,001]", "Cost-complexity pruning strength"],
        ],
        col_widths=[1.8, 2.0, 2.5]
    )
    add_body(doc,
        "Kết quả: thời gian chạy 167,5 giây. Bộ tham số tốt nhất:"
    )
    add_table(doc,
        headers=["Siêu tham số", "Giá trị tốt nhất"],
        rows=[
            ["n_estimators", "100"],
            ["max_depth", "None (không giới hạn)"],
            ["min_samples_split", "5"],
            ["ccp_alpha", "0,0"],
            ["class_weight", "'balanced'"],
        ],
        col_widths=[2.5, 2.5]
    )
    add_body(doc,
        "F1 trên 5-fold CV (train): 0,9445 — cải thiện +0,0011 so với baseline. "
        "Nhận xét: RandomizedSearch với 20 iteration đã tìm được tham số gần tối "
        "ưu trong thời gian hợp lý. Việc max_depth=None (không cắt) được chọn "
        "cho thấy RF vẫn hưởng lợi từ cây sâu khi kết hợp với min_samples_split=5 "
        "— pruning nhẹ ở mức split thay vì depth."
    )

    add_heading(doc, "4.2. Tinh chỉnh SVM", level=2)
    add_body(doc,
        "SVM kernel RBF trên 26.000 mẫu với GridSearchCV đầy đủ sẽ tốn nhiều "
        "giờ chạy (độ phức tạp O(n²)–O(n³)). Do đó áp dụng chiến lược tối ưu "
        "chi phí:"
    )
    add_bullet(doc,
        "Tìm kiếm trên subsample 10.000 mẫu stratified (pass rate giữ nguyên "
        "47,2%) thay vì toàn bộ 26.074 mẫu."
    )
    add_bullet(doc,
        "Dùng RandomizedSearchCV(n_iter=20, cv=3) thay vì GridSearchCV — "
        "giảm số lần fit từ hàng nghìn xuống còn 60 (20 × 3 fold)."
    )
    add_body(doc,
        "Không gian tìm kiếm:"
    )
    add_table(doc,
        headers=["Siêu tham số", "Dải giá trị"],
        rows=[
            ["C", "[0,1; 1; 10; 100]"],
            ["kernel", "['linear', 'rbf', 'poly']"],
            ["gamma", "['scale', 'auto', 0,01, 0,1]"],
        ],
        col_widths=[2.0, 4.0]
    )
    add_body(doc,
        "Kết quả: thời gian chạy 419,4 giây (subsample 10k). Bộ tham số tốt nhất:"
    )
    add_table(doc,
        headers=["Siêu tham số", "Giá trị tốt nhất"],
        rows=[
            ["kernel", "linear"],
            ["C", "100,0"],
            ["gamma", "auto (không quan trọng với linear kernel)"],
        ],
        col_widths=[2.5, 3.5]
    )
    add_body(doc,
        "F1 trên subsample (cv=3): 0,9352. Nhận xét đáng chú ý: kernel linear "
        "thắng kernel RBF — điều này gợi ý biên phân lớp tuyến tính đủ để phân "
        "tách dữ liệu OULAD sau khi đã qua OHE + StandardScaler. "
        "C=100 (regularisation yếu) cho thấy dữ liệu gần tuyến tính tách được, "
        "không cần margin rộng."
    )

    add_figure(doc, "03_hyperparam_top5.png",
               "Hình 5: Top-5 bộ tham số tốt nhất từ RandomizedSearchCV "
               "cho RF (trái) và SVM (phải) — mean CV F1 ± std")

    # =========================================================================
    # 5. SO SÁNH MÔ HÌNH CUỐI VÀ ĐÁNH GIÁ TRÊN TẬP TEST
    # =========================================================================
    add_heading(doc,
        "5. So sánh Mô hình Cuối cùng và Đánh giá trên Tập kiểm tra",
        level=1
    )

    add_heading(doc, "5.1. Champion RF vs Champion SVM — 5-Fold CV", level=2)
    add_body(doc,
        "Sau khi tinh chỉnh, hai mô hình tốt nhất (champion) được đánh giá lại "
        "đầy đủ trên toàn bộ tập train (26.074 mẫu) với 5-fold CV:"
    )
    add_table(doc,
        headers=["Mô hình", "F1 ± std", "AUC ± std", "Accuracy ± std",
                 "Overfit gap", "Fit time (s/fold)"],
        rows=[
            ["Champion RF\n(n_est=100, split=5, depth=None)",
             "0,9436 ± 0,0028", "0,9845 ± 0,0011", "0,9450 ± 0,0027",
             "+0,0478", "4,97"],
            ["Champion SVM\n(linear, C=100)",
             "0,9349 ± 0,0021", "0,9811 ± 0,0015", "0,9357 ± 0,0020",
             "+0,0012", "3.222,87"],
        ],
        col_widths=[1.8, 1.5, 1.5, 1.5, 1.0, 1.5]
    )
    add_body(doc,
        "So sánh với baseline:"
    )
    add_bullet(doc,
        "Champion RF cải thiện nhẹ so với RF baseline (F1: +0,0002, "
        "overfit gap giảm từ 0,0551 xuống 0,0478), nhưng tốc độ fit "
        "tăng từ 3,46 → 4,97 s/fold do n_estimators giữ nguyên 100 "
        "nhưng min_samples_split=5 thêm kiểm tra."
    )
    add_bullet(doc,
        "Champion SVM (linear C=100) chậm hơn rất nhiều so với SVM baseline "
        "RBF: 3.222,87 s/fold (~54 phút tổng 5 fold) vs 168,29 s/fold. "
        "Kernel linear trên 26.000 mẫu vẫn tốn kém do n mẫu lớn và C=100 "
        "yêu cầu biên hẹp với nhiều support vectors."
    )

    add_figure(doc, "03_champion_compare.png",
               "Hình 6: So sánh Champion RF và Champion SVM — "
               "F1 theo từng fold với khoảng tin cậy ±std")

    add_heading(doc, "5.2. Đường học (Learning Curves)", level=2)
    add_body(doc,
        "Learning curve được vẽ để phân tích hành vi bias/variance và khả năng "
        "hưởng lợi từ dữ liệu bổ sung của từng thuật toán:"
    )
    add_body(doc,
        "RF learning curve (toàn bộ tập train, train_sizes từ 10% đến 67%):"
    )
    add_table(doc,
        headers=["Kích thước train", "RF Train F1", "RF CV F1", "Gap"],
        rows=[
            ["1.738 (10%)",  "0,9924", "0,9346", "0,0578"],
            ["4.866 (28%)",  "0,9933", "0,9389", "0,0544"],
            ["7.995 (46%)",  "0,9933", "0,9412", "0,0522"],
            ["11.124 (64%)", "0,9931", "0,9421", "0,0510"],
            ["14.253 (82%)", "0,9926", "0,9438", "0,0488"],
            ["17.382 (100%)", "0,9927", "0,9444", "0,0483"],
        ],
        col_widths=[1.7, 1.5, 1.5, 1.1]
    )
    add_body(doc,
        "SVM learning curve (subsample 8.000 mẫu, train_sizes từ 10% đến 67%):"
    )
    add_table(doc,
        headers=["Kích thước train", "SVM Train F1", "SVM CV F1", "Gap"],
        rows=[
            ["533 (10%)",   "0,9565", "0,9081", "0,0484"],
            ["1.493 (28%)", "0,9433", "0,9286", "0,0147"],
            ["2.453 (46%)", "0,9370", "0,9292", "0,0078"],
            ["3.413 (64%)", "0,9333", "0,9309", "0,0024"],
            ["4.373 (82%)", "0,9342", "0,9311", "0,0031"],
            ["5.333 (100%)", "0,9348", "0,9319", "0,0029"],
        ],
        col_widths=[1.7, 1.5, 1.5, 1.1]
    )
    add_figure(doc, "03_learning_curves.png",
               "Hình 7: Learning Curve — RF (trái) và SVM (phải). "
               "Đường liền: Train F1; Đường đứt: CV F1; Vùng tô: ±std")
    add_body(doc,
        "Phân tích Learning Curve:"
    )
    add_bullet(doc,
        "RF: Train F1 bão hoà ngay từ sớm (~0,993), gap với CV F1 lớn "
        "(0,048–0,058) và thu hẹp chậm khi thêm dữ liệu — đặc trưng của "
        "High Variance (overfit). Mô hình có thể hưởng lợi từ thêm dữ liệu "
        "nhưng giới hạn chính là phương sai của bản thân thuật toán."
    )
    add_bullet(doc,
        "SVM (linear): Gap thu hẹp rất nhanh từ 0,048 (10% dữ liệu) xuống "
        "~0,003 (từ 50% trở đi) — đường Train và CV hội tụ gần nhau. "
        "Điều này xác nhận SVM linear bị High Bias nhẹ (underfitting nhẹ) "
        "nhưng tổng quát hoá tốt hơn RF. Thêm dữ liệu không cải thiện "
        "đáng kể sau ngưỡng ~50%."
    )

    add_heading(doc, "5.3. Đánh giá trên Tập kiểm tra (Test Set — 20% Hold-out)", level=2)
    add_body(doc,
        "Cả hai champion được refit trên toàn bộ tập train (26.074 mẫu) "
        "và dự đoán trên tập test (6.519 mẫu) giữ kín từ đầu:"
    )
    add_table(doc,
        headers=["Mô hình", "F1", "AUC", "Accuracy", "Precision", "Recall",
                 "Fit time (s)", "Infer time (s)"],
        rows=[
            ["Champion RF",  "0,9486", "0,9859", "0,9501", "0,9231", "0,9756",
             "1,60", "0,65"],
            ["Champion SVM", "0,9392", "0,9843", "0,9403", "0,9041", "0,9773",
             "4.924,72", "0,72"],
        ],
        col_widths=[1.4, 0.7, 0.7, 0.8, 0.9, 0.8, 1.0, 1.0]
    )
    add_body(doc,
        "Classification Report — Champion RF (tập test):"
    )
    add_table(doc,
        headers=["Lớp", "Precision", "Recall", "F1-score", "Support"],
        rows=[
            ["Fail/Withdrawn (0)", "0,977", "0,927", "0,952", "3.442"],
            ["Pass/Distinction (1)", "0,923", "0,976", "0,949", "3.077"],
            ["Accuracy", "", "", "0,950", "6.519"],
            ["Macro avg", "0,950", "0,952", "0,950", "6.519"],
        ],
        col_widths=[2.0, 1.2, 1.2, 1.2, 1.2]
    )
    add_body(doc,
        "Classification Report — Champion SVM (tập test):"
    )
    add_table(doc,
        headers=["Lớp", "Precision", "Recall", "F1-score", "Support"],
        rows=[
            ["Fail/Withdrawn (0)", "0,978", "0,907", "0,941", "3.442"],
            ["Pass/Distinction (1)", "0,904", "0,977", "0,939", "3.077"],
            ["Accuracy", "", "", "0,940", "6.519"],
            ["Macro avg", "0,941", "0,942", "0,940", "6.519"],
        ],
        col_widths=[2.0, 1.2, 1.2, 1.2, 1.2]
    )

    add_heading(doc, "5.4. Quyết định Chọn Mô hình", level=2)
    add_body(doc,
        "Dựa trên đánh giá toàn diện, mô hình được chọn là Champion Random "
        "Forest với lý do:"
    )
    add_table(doc,
        headers=["Tiêu chí", "Champion RF", "Champion SVM", "Lợi thế"],
        rows=[
            ["F1 (test)", "0,9486", "0,9392", "RF (+0,94 pp)"],
            ["AUC (test)", "0,9859", "0,9843", "RF (+0,16 pp)"],
            ["Accuracy (test)", "0,9501", "0,9403", "RF (+0,98 pp)"],
            ["Precision (test)", "0,9231", "0,9041", "RF (+1,90 pp)"],
            ["Recall (test)", "0,9756", "0,9773", "SVM (+0,17 pp — xấp xỉ bằng nhau)"],
            ["Overfit gap (CV)", "0,0478", "0,0012", "SVM (tổng quát hoá tốt hơn)"],
            ["Fit time (refit)", "1,60 s", "4.924,72 s", "RF (×3.078 lần nhanh hơn)"],
            ["Inference (6.519 mẫu)", "0,65 s", "0,72 s", "RF (xấp xỉ bằng nhau)"],
            ["Stability (F1 std CV)", "±0,0028", "±0,0021", "SVM (ổn định hơn nhẹ)"],
        ],
        col_widths=[2.0, 1.2, 1.2, 2.0]
    )
    add_body(doc,
        "RF vượt trội về F1 (+0,94pp), Accuracy (+0,98pp) và đặc biệt về "
        "tốc độ huấn luyện (hơn 3.000 lần). Trong khi SVM có overfit gap "
        "thấp hơn và ổn định hơn nhẹ giữa các fold, hai yếu tố này không "
        "đủ bù đắp cho hiệu năng dự đoán và chi phí tính toán vượt trội của RF. "
        "Kết luận: Champion Random Forest là mô hình sản xuất."
    )

    # =========================================================================
    # 6. PHÂN TÍCH LỖI CHUYÊN SÂU
    # =========================================================================
    add_heading(doc, "6. Phân tích Lỗi Chuyên sâu (Deep Error Analysis)", level=1)

    add_heading(doc, "6.1. Ma trận Nhầm lẫn (Confusion Matrix)", level=2)
    add_figure(doc, "03_confusion_matrices.png",
               "Hình 8: Ma trận nhầm lẫn trên tập test — Champion RF (trái) "
               "và Champion SVM (phải)")
    add_body(doc,
        "Từ classification report và confusion matrix trên tập test:"
    )
    add_table(doc,
        headers=["Loại lỗi", "RF", "SVM", "Nhận xét"],
        rows=[
            ["False Positive (FP)\ndự đoán Pass, thực tế Fail",
             "254 (7,4%)", "330 (9,6%)", "RF ít FP hơn — Precision cao hơn"],
            ["False Negative (FN)\ndự đoán Fail, thực tế Pass",
             "75 (2,4%)", "71 (2,3%)", "Xấp xỉ bằng nhau — Recall tương đương"],
        ],
        col_widths=[2.5, 0.8, 0.8, 2.3]
    )
    add_body(doc,
        "RF mắc ít lỗi False Positive hơn — quan trọng trong bài toán cảnh "
        "báo rủi ro học tập: ít báo động giả sẽ giữ được niềm tin của giáo "
        "viên và sinh viên vào hệ thống dự đoán."
    )

    add_heading(doc, "6.2. Phân tích Bất đồng Dự đoán (Disagreement Analysis)", level=2)
    add_body(doc,
        "Phân chia 6.519 mẫu test thành 4 nhóm theo sự đồng thuận / bất "
        "đồng giữa hai mô hình:"
    )
    add_table(doc,
        headers=["Nhóm", "Số mẫu", "Tỉ lệ"],
        rows=[
            ["Cả 2 đúng", "6.066", "93,05%"],
            ["Cả 2 sai", "261", "4,00%"],
            ["Chỉ RF đúng", "128", "1,96%"],
            ["Chỉ SVM đúng", "64", "0,98%"],
        ],
        col_widths=[3.0, 1.5, 1.5]
    )
    add_body(doc,
        "93% mẫu được cả hai mô hình dự đoán đúng. Nhóm 'Chỉ RF đúng' "
        "(128 mẫu) gấp đôi nhóm 'Chỉ SVM đúng' (64 mẫu) — RF có lợi thế "
        "rõ ràng trên các mẫu 'khó'."
    )
    add_body(doc,
        "Phân tích đặc trưng theo nhóm (trung bình trên tập numeric):"
    )
    add_table(doc,
        headers=["Đặc trưng", "Cả 2 đúng", "Cả 2 sai", "Chỉ RF đúng", "Chỉ SVM đúng"],
        rows=[
            ["total_clicks",          "1.207",  "1.305",  "1.387",  "1.076"],
            ["mean_score",            "71,3",   "63,0",   "55,0",   "57,8"],
            ["num_active_days",       "54,9",   "63,4",   "67,5",   "55,2"],
            ["mean_days_early",       "12,6",   "9,7",    "6,2",    "5,6"],
            ["num_late_submissions",  "1,40",   "2,90",   "2,45",   "3,64"],
            ["pct_tma_submitted",     "0,51",   "0,61",   "0,58",   "0,62"],
        ],
        col_widths=[2.2, 1.2, 1.1, 1.2, 1.2]
    )
    add_body(doc,
        "Quan sát:"
    )
    add_bullet(doc,
        "Nhóm 'Cả 2 sai' có mean_score thấp (63,0), num_active_days cao "
        "(63,4) — sinh viên hoạt động nhiều nhưng điểm thấp, khó phân loại "
        "với cả hai thuật toán."
    )
    add_bullet(doc,
        "Nhóm 'Chỉ RF đúng' có mean_score rất thấp (55,0) nhưng "
        "total_clicks cao (1.387) — RF phát hiện pattern phi tuyến giữa "
        "điểm thấp và hoạt động cao, trong khi SVM linear bỏ sót."
    )
    add_bullet(doc,
        "Nhóm 'Chỉ SVM đúng' có total_clicks thấp (1.076) và "
        "num_late_submissions cao (3,64) — SVM nhạy hơn với biên tuyến tính "
        "trên các đặc trưng hành vi giao nộp muộn."
    )

    add_heading(doc, "6.3. Đường cong ROC (ROC Curve)", level=2)
    add_figure(doc, "03_roc_curves.png",
               "Hình 9: ROC Curve trên tập test — RF (AUC=0,986) và "
               "SVM (AUC=0,984)")
    add_body(doc,
        "Cả hai mô hình có AUC > 0,98 — khả năng phân tách hai lớp rất tốt "
        "ở mọi ngưỡng quyết định. RF có AUC cao hơn SVM +0,0016 — khoảng "
        "cách nhỏ nhưng nhất quán với các độ đo khác. Với ứng dụng cảnh báo "
        "sớm, có thể hạ ngưỡng dự đoán xuống dưới 0,5 để tăng Recall "
        "(phát hiện thêm sinh viên có nguy cơ trượt) mà vẫn giữ AUC cao."
    )

    add_heading(doc, "6.4. Phân phối Xác suất Dự đoán", level=2)
    add_figure(doc, "03_prob_histogram.png",
               "Hình 10: Phân phối xác suất dự đoán P(Pass) của RF và SVM "
               "theo nhãn thực tế")
    add_body(doc,
        "Biểu đồ histogram xác suất cho thấy cả hai mô hình phân tách rõ "
        "hai lớp: nhãn 0 (Fail) tập trung ở xác suất thấp (<0,3) và nhãn 1 "
        "(Pass) tập trung ở xác suất cao (>0,7). Vùng chồng lấp "
        "(0,3–0,7) ít — xác nhận mô hình có độ tin cậy cao."
    )

    add_heading(doc, "6.5. Ranh giới Quyết định trong Không gian PCA 2D", level=2)
    add_figure(doc, "03_pca_decision_boundary.png",
               "Hình 11: Ranh giới quyết định của RF (trái) và SVM linear "
               "(phải) trong không gian PCA 2D (2 thành phần giải thích 17,8% "
               "phương sai)")
    add_body(doc,
        "Hai thành phần PCA giải thích 17,8% phương sai tổng — đủ để minh "
        "hoạ trực quan nhưng không đại diện đầy đủ cho không gian 47 chiều. "
        "Ranh giới của RF (trái) phân mảnh theo hình chữ nhật trục song song "
        "(axis-aligned splits) — đặc trưng của cây quyết định. Ranh giới của "
        "SVM linear (phải) mượt hơn, gần với hyperplane tuyến tính — phù hợp "
        "với kernel linear đã chọn qua tuning."
    )

    add_heading(doc, "6.6. Thảo luận — Nguyên nhân Lỗi và Đề xuất Cải thiện", level=2)
    add_body(doc, "Nguyên nhân lỗi chính:", bold=True)
    add_bullet(doc,
        "Thiếu đặc trưng thời gian (temporal): các đặc trưng như tổng click, "
        "mean_score là thống kê tĩnh toàn khoá học — không nắm bắt xu hướng "
        "thay đổi theo tuần (sinh viên mất hứng giữa kỳ, bỏ cuối kỳ). Mô "
        "hình sequence-aware (LSTM, Transformer) trên chuỗi VLE theo tuần "
        "có thể cải thiện đáng kể."
    )
    add_bullet(doc,
        "Nhiễu nhãn: một số bản ghi 'Withdrawn' giai đoạn cuối khoá (gần "
        "hoàn thành) có đặc trưng rất giống Pass nhưng nhãn là Fail — không "
        "thể xử lý chỉ bằng thuật toán mà cần làm sạch nhãn."
    )
    add_bullet(doc,
        "Thiếu thông tin ngoài hệ thống: hoàn cảnh cá nhân (việc làm, sức "
        "khoẻ, hỗ trợ tài chính) quyết định nhiều đến kết quả học nhưng "
        "không có trong bộ dữ liệu OULAD."
    )
    add_body(doc, "Đề xuất cải thiện:", bold=True)
    add_bullet(doc,
        "Feature engineering động học: tốc độ giảm weekly_clicks, khoảng "
        "cách từ lần tương tác cuối đến deadline, số tuần liên tiếp không "
        "đăng nhập VLE."
    )
    add_bullet(doc,
        "Stacking Ensemble: kết hợp RF và SVM (hoặc LightGBM) qua "
        "StackingClassifier với meta-learner LogisticRegression — tận dụng "
        "điểm mạnh khác nhau của hai họ mô hình (RF phát hiện tương tác "
        "phi tuyến, SVM linear với biên chắc chắn)."
    )
    add_bullet(doc,
        "Hiệu chỉnh ngưỡng quyết định: hạ ngưỡng từ 0,5 xuống 0,3–0,4 "
        "để ưu tiên Recall cao hơn (phát hiện sớm sinh viên có nguy cơ), "
        "phục vụ can thiệp kịp thời — đánh đổi giảm Precision nhẹ."
    )

    # =========================================================================
    # Kết luận phần
    # =========================================================================
    add_heading(doc, "Kết luận Phần III", level=1)
    add_body(doc,
        "Báo cáo đã trình bày quá trình huấn luyện và so sánh toàn diện "
        "hai thuật toán Random Forest và SVM trên bộ dữ liệu OULAD. "
        "Kết quả thực nghiệm cho thấy:"
    )
    add_bullet(doc,
        "Cả hai thuật toán đạt hiệu năng cao (F1 > 0,93) nhờ chất lượng "
        "dữ liệu tốt sau tiền xử lý."
    )
    add_bullet(doc,
        "Random Forest vượt trội về F1 (+0,94pp trên test), AUC (+0,16pp) "
        "và đặc biệt về tốc độ huấn luyện (hơn 3.000 lần nhanh hơn SVM "
        "linear trên 26.000 mẫu)."
    )
    add_bullet(doc,
        "SVM có ưu điểm overfit thấp hơn (gap chỉ 0,001) và ranh giới "
        "quyết định mượt — phù hợp khi dữ liệu tuyến tính tách được và "
        "tài nguyên tính toán không là ràng buộc."
    )
    add_bullet(doc,
        "Mô hình được chọn cho production: Champion Random Forest "
        "(F1_test=0,9486, AUC_test=0,9859, fit_time=1,6s)."
    )

    return doc


def main():
    doc = build_document()
    out = ROOT / "reports" / "BaoCao_HuanLuyen_RF_SVM.docx"
    doc.save(str(out))
    print("Saved:", str(out).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
