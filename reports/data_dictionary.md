# Mô tả dữ liệu OULAD

Tài liệu này mô tả 7 bảng trong [data/raw/](../data/raw/) của bộ dữ liệu **OULAD (Open University Learning Analytics Dataset)** — dữ liệu học tập trực tuyến của Open University (UK), dùng để dự đoán kết quả học tập của sinh viên.

## Quy ước chung

- **Đơn vị thời gian:** hầu hết các trường thời gian (`date`, `date_submitted`, `date_registration`, ...) tính bằng **số ngày tương đối so với ngày bắt đầu của presentation** (ngày 0). Giá trị âm = trước khi presentation bắt đầu, dương = sau khi presentation bắt đầu.
- **Code presentation:** mã kỳ khai giảng theo định dạng `YYYY` + ký tự (`B` = February, `J` = October). Ví dụ `2013J` = kỳ tháng 10/2013, `2014B` = kỳ tháng 2/2014.
- **Code module:** mã môn học, ẩn danh (`AAA`, `BBB`, `CCC`, ...).
- **Granularity của bài toán:** một sinh viên có thể đăng ký cùng một môn ở nhiều presentation khác nhau, hoặc đăng ký nhiều môn. Khóa logic xuyên suốt là **(`id_student`, `code_module`, `code_presentation`)**.

## Sơ đồ quan hệ

```
courses (code_module, code_presentation)
   │
   ├── studentInfo ──┬── studentRegistration
   │                 │
   │                 ├── studentAssessment ── assessments
   │                 │                            │
   │                 └── studentVle ── vle ──────┘
   │                          (id_site)
   └── (gốc cho mọi bảng theo presentation)
```

---

## 1. `studentInfo.csv` — Thông tin sinh viên

**Tổng quan:** Bảng trung tâm chứa thông tin nhân khẩu học (demographics) và **nhãn kết quả cuối kỳ** (`final_result`) của mỗi sinh viên trong từng presentation. Đây là **base table** cho bài toán phân loại Pass/Fail — mỗi dòng = một (sinh viên × môn × kỳ).

**Số dòng:** 32,593
**Khóa chính:** `(id_student, code_module, code_presentation)`

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `code_module` | string | — | Mã môn học (AAA, BBB, ...) |
| `code_presentation` | string | — | Mã kỳ khai giảng (2013J, 2014B, ...) |
| `id_student` | int | — | ID sinh viên (đã ẩn danh) |
| `gender` | string | — | Giới tính: `M` (Male), `F` (Female) |
| `region` | string | — | Vùng địa lý của sinh viên ở UK (East Anglian Region, Scotland, ...) |
| `highest_education` | string | — | Trình độ học vấn cao nhất trước khi vào khóa: `No Formal quals`, `Lower Than A Level`, `A Level or Equivalent`, `HE Qualification`, `Post Graduate Qualification` |
| `imd_band` | string | %, decile | Index of Multiple Deprivation — phân vị mức độ thiếu thốn của khu vực sinh sống, dạng `0-10%`, `10-20`, ..., `90-100%`. Có thể NaN nếu sinh viên không khai báo. |
| `age_band` | string | tuổi | Nhóm tuổi: `0-35`, `35-55`, `55<=` |
| `num_of_prev_attempts` | int | lần | Số lần sinh viên đã học môn này trước đó (0 = lần đầu) |
| `studied_credits` | int | tín chỉ | Tổng số tín chỉ sinh viên đăng ký trong kỳ này (qua tất cả các môn) |
| `disability` | string | — | Có khai báo khuyết tật không: `Y` / `N` |
| `final_result` | string | — | **Nhãn**: `Pass`, `Distinction`, `Fail`, `Withdrawn` |

---

## 2. `studentRegistration.csv` — Đăng ký môn học

**Tổng quan:** Ghi nhận thời điểm đăng ký và (nếu có) thời điểm hủy đăng ký của sinh viên cho từng presentation. Sinh viên `Withdrawn` thường có `date_unregistration` không null.

**Số dòng:** 32,593
**Khóa chính:** `(id_student, code_module, code_presentation)` (1-1 với `studentInfo`)

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `code_module` | string | — | Mã môn học |
| `code_presentation` | string | — | Mã kỳ khai giảng |
| `id_student` | int | — | ID sinh viên |
| `date_registration` | int | ngày | Ngày đăng ký, tính theo ngày tương đối so với ngày bắt đầu presentation. Hầu như luôn âm (đăng ký TRƯỚC khi presentation bắt đầu). |
| `date_unregistration` | int / null | ngày | Ngày hủy đăng ký (nếu có), cùng quy ước. **NaN = sinh viên không hủy.** ⚠️ Trường này **leak** thông tin Withdrawn → cẩn trọng khi dùng làm feature. |

---

## 3. `studentAssessment.csv` — Kết quả từng bài đánh giá

**Tổng quan:** Mỗi dòng = một lần sinh viên nộp một assessment (TMA, CMA, hoặc Exam). Lưu điểm và thời điểm nộp. Sinh viên không nộp → không xuất hiện trong bảng này.

**Số dòng:** 173,912
**Khóa chính:** `(id_student, id_assessment)`
**Quan hệ:** join với [assessments.csv](#4-assessmentscsv--metadata-bài-đánh-giá) qua `id_assessment` để lấy `code_module`, `code_presentation`, `weight`, `assessment_type`, `date` (deadline).

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `id_assessment` | int | — | ID bài đánh giá (FK → `assessments.id_assessment`) |
| `id_student` | int | — | ID sinh viên |
| `date_submitted` | int | ngày | Ngày sinh viên nộp bài, tính theo quy ước chung |
| `is_banked` | int (0/1) | boolean | Có chuyển kết quả từ presentation cũ sang không (1 = banked, 0 = nộp mới) |
| `score` | float | điểm | Điểm số bài làm, thang **0–100**. Có thể null/`?` (rất ít) — nên ép `pd.to_numeric(errors='coerce')` |

**Suy luận trễ hạn:** `is_late = date_submitted > assessments.date`.

---

## 4. `assessments.csv` — Metadata bài đánh giá

**Tổng quan:** Mô tả từng assessment trong mỗi presentation: loại (TMA / CMA / Exam), trọng số, hạn nộp.

**Số dòng:** 206
**Khóa chính:** `id_assessment`

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `code_module` | string | — | Mã môn học |
| `code_presentation` | string | — | Mã kỳ khai giảng |
| `id_assessment` | int | — | ID bài đánh giá (PK) |
| `assessment_type` | string | — | Loại: `TMA` (Tutor-Marked Assessment), `CMA` (Computer-Marked Assessment), `Exam` (cuối kỳ) |
| `date` | int / null | ngày | Hạn nộp, tính theo quy ước chung. `Exam` có thể null nếu lịch thi chưa cố định. |
| `weight` | float | % | Trọng số đóng góp vào điểm cuối kỳ. Tổng các CMA + TMA ≈ 100%; `Exam` thường có `weight = 100` (tính riêng). |

---

## 5. `studentVle.csv` — Click log trên môi trường học (VLE)

**Tổng quan:** Log tương tác hằng ngày của sinh viên với từng resource trong VLE (Virtual Learning Environment). Đây là bảng **lớn nhất** — granularity rất chi tiết: 1 dòng = 1 sinh viên × 1 resource × 1 ngày. Cần aggregate trước khi join.

**Số dòng:** 10,655,280
**Khóa logic:** `(id_student, code_module, code_presentation, id_site, date)`
**Quan hệ:** join `vle.csv` qua `id_site` để lấy `activity_type`.

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `code_module` | string | — | Mã môn học |
| `code_presentation` | string | — | Mã kỳ khai giảng |
| `id_student` | int | — | ID sinh viên |
| `id_site` | int | — | ID resource trong VLE (FK → `vle.id_site`) |
| `date` | int | ngày | Ngày phát sinh tương tác, theo quy ước chung. Có thể âm (sinh viên xem trước khi presentation bắt đầu). |
| `sum_click` | int | lần click | Số lần click trong ngày đó vào resource đó |

**Tip RAM:** dùng `dtype={'id_student':'int32', 'id_site':'int32', 'date':'int16', 'sum_click':'int32'}` khi `read_csv` để tiết kiệm bộ nhớ (~10M dòng).

---

## 6. `vle.csv` — Metadata resource VLE

**Tổng quan:** Mô tả từng resource (trang nội dung, video, forum, ...) trong VLE.

**Số dòng:** 6,364
**Khóa chính:** `id_site`

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `id_site` | int | — | ID resource (PK) |
| `code_module` | string | — | Mã môn học |
| `code_presentation` | string | — | Mã kỳ khai giảng |
| `activity_type` | string | — | Loại resource: `resource`, `oucontent`, `url`, `forumng`, `quiz`, `subpage`, `homepage`, `glossary`, ... (~20 loại) |
| `week_from` | int / null | tuần | Tuần bắt đầu resource có hiệu lực (1-indexed theo presentation). NaN = không xác định. |
| `week_to` | int / null | tuần | Tuần kết thúc. NaN = không xác định. |

---

## 7. `courses.csv` — Metadata presentation

**Tổng quan:** Bảng nhỏ nhất — chỉ chứa độ dài (số ngày) của mỗi presentation. Hữu ích để chuẩn hóa các feature thời gian theo độ dài kỳ học.

**Số dòng:** 22
**Khóa chính:** `(code_module, code_presentation)`

| Trường | Kiểu | Đơn vị | Ý nghĩa |
| --- | --- | --- | --- |
| `code_module` | string | — | Mã môn học |
| `code_presentation` | string | — | Mã kỳ khai giảng |
| `module_presentation_length` | int | ngày | Tổng số ngày của presentation (thường ~240–270 ngày) |

---

## Tham khảo

- Bộ dữ liệu gốc: <https://analyse.kmi.open.ac.uk/open_dataset>
- Paper: Kuzilek, Hlosta & Zdrahal (2017), *Open University Learning Analytics Dataset*, Scientific Data.
- Notebook xử lý: [notebooks/01_eda_preprocessing.ipynb](../notebooks/01_eda_preprocessing.ipynb)
