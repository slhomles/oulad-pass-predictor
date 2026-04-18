from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import predict, upload, training
from app.core.scheduler import start_scheduler, shutdown_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(
    title="Predict Pass Student API",
    description="API dự đoán học sinh pass/fail dựa trên OULAD + Random Forest.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(training.router, prefix="/api", tags=["training"])


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
