from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ScheduleConfig(BaseModel):
    cron: str


@router.post("/train")
def train_now():
    # TODO: gọi app.ml.train.run()
    raise NotImplementedError


@router.get("/schedule")
def get_schedule():
    # TODO: trả cron hiện tại + next run time
    raise NotImplementedError


@router.post("/schedule")
def update_schedule(cfg: ScheduleConfig):
    # TODO: reschedule APScheduler job với cfg.cron
    raise NotImplementedError
