from pydantic import BaseModel


class StudentFeatures(BaseModel):
    gender: str
    region: str
    highest_education: str
    imd_band: str | None = None
    age_band: str
    num_of_prev_attempts: int
    studied_credits: int
    disability: str
    sum_click: int = 0


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
