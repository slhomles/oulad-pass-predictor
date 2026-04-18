from fastapi import APIRouter

from app.schemas.student import StudentFeatures, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    # TODO: gọi app.ml.predict.run(features)
    raise NotImplementedError
