# Frontend

Scaffold bằng Vite sau khi folder structure được tạo:

```bash
cd frontend
npm create vite@latest . -- --template react
npm install axios react-router-dom
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Sau đó thêm:
- `src/components/PredictForm.jsx` — form dự đoán
- `src/components/DataUpload.jsx` — upload CSV
- `src/components/TrainingPanel.jsx` — trigger + schedule training
- `src/services/api.js` — axios wrapper, đọc `VITE_API_BASE_URL`
