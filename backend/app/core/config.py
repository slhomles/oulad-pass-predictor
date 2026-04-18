from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATA_DIR: Path = Path("./data")
    MODEL_DIR: Path = Path("./models")
    TRAIN_SCHEDULE_CRON: str = "0 0 * * 0"

    class Config:
        env_file = ".env"


settings = Settings()
