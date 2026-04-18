from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings

scheduler = BackgroundScheduler()
JOB_ID = "retrain_model"


def _retrain():
    # Lazy import tránh vòng phụ thuộc
    from app.ml.train import run
    run()


def start_scheduler():
    trigger = CronTrigger.from_crontab(settings.TRAIN_SCHEDULE_CRON)
    scheduler.add_job(_retrain, trigger=trigger, id=JOB_ID, replace_existing=True)
    scheduler.start()


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)


def reschedule(cron: str):
    scheduler.reschedule_job(JOB_ID, trigger=CronTrigger.from_crontab(cron))
