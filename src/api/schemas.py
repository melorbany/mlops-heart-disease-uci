from pydantic import BaseModel


class HeartFeatures(BaseModel):
    # Adjust types accordingly to your dataset
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
