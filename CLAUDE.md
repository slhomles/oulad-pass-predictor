# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Predict Pass Student** — dự đoán kết quả pass/fail của sinh viên dựa trên dataset [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) bằng Random Forest. Đây là project học thuật cho môn Data Warehouse and Data Mining (Kỳ 2 Năm 4).

## Commands

### Backend

```bash
cd backend
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload    # chạy tại http://localhost:8000
```

Run tests:
```bash
cd backend
pytest tests/
pytest tests/test_api.py::test_root   # chạy một test cụ thể
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # chạy tại http://localhost:5173
```

### Docker (toàn bộ stack)

```bash
docker-compose up --build
```

### Notebooks (EDA)

```bash
pip install -r backend/requirements.txt jupyter
jupyter notebook notebooks/
```

## Architecture

```
React (Vite)  →  FastAPI backend  →  models/rf_latest.pkl
                      │
                  APScheduler (cron retrain)
                      │
              data/raw/*.csv  →  preprocessing  →  train
```

**Backend** (`backend/app/`):
- `main.py` — khởi tạo FastAPI, CORS, gắn routers, start/shutdown APScheduler qua `lifespan`.
- `core/config.py` — đọc cấu hình từ `.env` (DATA_DIR, MODEL_DIR, TRAIN_SCHEDULE_CRON).
- `core/scheduler.py` — APScheduler BackgroundScheduler, job ID `retrain_model`, hỗ trợ `reschedule()` động.
- `ml/preprocessing.py` — **chưa implement**: cần `load_raw()` và `build_features()`. Chiến lược: gộp `studentVle` (10M+ rows) trước bằng `groupby` → join với `studentInfo` → impute NaN click = 0 → Label Encode categoricals (Random Forest không cần scaling). Top 20 features theo Gini importance.
- `ml/train.py` — train RandomForestClassifier(n_estimators=200), lưu `rf_<timestamp>.pkl` + `rf_latest.pkl`.
- `ml/predict.py` — load `rf_latest.pkl` qua `@lru_cache`; **chưa implement**: cần chuyển `StudentFeatures` → DataFrame đúng schema trước khi gọi `predict_proba`.
- `api/predict.py`, `api/upload.py`, `api/training.py` — **toàn bộ TODO**: cần implement gọi lớp `ml/`.
- `schemas/student.py` — Pydantic schema đầu vào: `StudentFeatures` (gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability, sum_click).

**Frontend** (`frontend/src/`):
- Hiện tại chỉ là Vite scaffold mặc định, chưa implement UI thực. Cần xây dựng form nhập `StudentFeatures` và gọi `VITE_API_BASE_URL=/api/predict`.

**Data flow**:
- CSV thô → `data/raw/` → `preprocessing.build_features()` → train → `models/rf_latest.pkl` → `predict.run()` → API response.

## Environment Variables

Copy `.env.example` → `.env`:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `APP_ENV` | `development` | |
| `DATA_DIR` | `./data` | thư mục chứa `raw/`, `processed/` |
| `MODEL_DIR` | `./models` | nơi lưu file `.pkl` |
| `TRAIN_SCHEDULE_CRON` | `0 0 * * 0` | cron UTC, mặc định Chủ Nhật 00:00 |
| `VITE_API_BASE_URL` | `http://localhost:8000/api` | dùng trong frontend |

## Dataset

Tải từ https://analyse.kmi.open.ac.uk/open_dataset, giải nén vào `data/raw/`. Các file cần thiết: `studentInfo.csv`, `studentVle.csv`, `assessments.csv`, `studentAssessment.csv`, `courses.csv`, `studentRegistration.csv`, `vle.csv`.

## Implementation Notes

- `studentVle` có 10M+ rows — luôn `groupby(['id_student', 'code_module', 'code_presentation'])` trước khi join để tránh OOM.
- Target: `final_result` in `{'Pass', 'Distinction'}` → 1, còn lại → 0.
- Categorical encoding dùng `LabelEncoder` hoặc `pd.get_dummies` (Random Forest không cần scaling số).
- Sau khi train lại, `ml/predict.py` dùng `@lru_cache` — cần `_load_model.cache_clear()` để load model mới.
- API docs tự sinh tại `/docs` (Swagger UI).
