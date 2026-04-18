# Predict Pass Student

Dự đoán học sinh **pass / fail** khóa học dựa trên dataset [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) bằng thuật toán **Random Forest**. Sản phẩm hoàn chỉnh gồm backend API, giao diện web, khả năng nạp thêm dữ liệu và lập lịch train lại tự động.

---

## Tính năng chính

- **Dự đoán**: nhập thông tin học sinh → trả về xác suất pass/fail.
- **Nạp thêm dữ liệu**: upload file CSV bổ sung vào tập training.
- **Huấn luyện theo lịch**: cấu hình cron để tự động train lại model định kỳ.
- **API docs**: Swagger UI tự sinh tại `/docs`.

---

## Kiến trúc

```
┌──────────────┐    HTTP     ┌──────────────────┐
│  React (UI)  │ ──────────► │  FastAPI backend │
└──────────────┘             │  ┌────────────┐  │
                             │  │ APScheduler│  │──► models/*.pkl
                             │  └────────────┘  │
                             └──────────────────┘
                                      │
                                      ▼
                              data/raw, data/processed
```

- **Backend**: FastAPI + scikit-learn + APScheduler.
- **Frontend**: React (Vite) + Tailwind.
- **Deploy**: Docker Compose.

---

## Cấu trúc thư mục

```
predict-pass-student/
├── backend/           # FastAPI + ML logic
├── frontend/          # React + Vite
├── notebooks/         # EDA, thử nghiệm model
├── data/              # raw + processed (gitignored)
├── models/            # model .pkl đã train (gitignored)
├── reports/figures/   # biểu đồ xuất từ notebook
├── docker-compose.yml
├── .gitignore
├── .env.example
└── README.md
```

---

## Cài đặt & chạy

### Option 1 — Docker (khuyến nghị)

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

### Option 2 — Chạy thủ công

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Notebook (EDA / nghiên cứu)

```bash
pip install -r backend/requirements.txt jupyter
jupyter notebook notebooks/
```

---

## Dataset

Tải OULAD từ: https://analyse.kmi.open.ac.uk/open_dataset

Giải nén các file `.csv` (studentInfo, studentVle, courses, assessments, ...) vào `data/raw/`.

---

## Kết quả model

| Metric    | Giá trị |
|-----------|---------|
| Accuracy  | _TBD_   |
| Precision | _TBD_   |
| Recall    | _TBD_   |
| F1        | _TBD_   |

(Sẽ cập nhật sau khi train xong.)

---

## Tác giả

- Phương — phuong6b2004@gmail.com
- Môn: Data Warehouse and Data Mining — Kỳ 2 Năm 4
