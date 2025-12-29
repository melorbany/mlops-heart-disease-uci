from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.config import FINAL_MODEL_PATH


class HeartDiseaseModelService:
    def __init__(self, model_path: Path = FINAL_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        return joblib.load(self.model_path)

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, float]:
        df = pd.DataFrame([features])
        prob = self.model.predict_proba(df)[:, 1][0]
        pred = int(prob >= 0.5)
        return {
            "prediction": pred,
            "probability": float(prob),
        }
