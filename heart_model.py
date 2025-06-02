from pathlib import Path
import joblib
import pandas as pd
from typing import Any, List, Union

FEATURES: List[str] = [
    "age",
    "cholesterol",
    "heart rate",
    "diabetes",
    "smoking",
    "alcohol consumption",
    "exercise hours per week",
    "stress level",
    "sedentary hours per day",
    "bmi",
    "physical activity days per week",
    "sleep hours per day",
    "troponin",
    "gender",
    "systolic blood pressure",
]

DEFAULT_MODEL_PATH = Path(__file__).with_name("best_DecisionTreeClassifier_model.pkl")

class HeartAttackModel:
    def __init__(self) -> None:
        self._pipeline = joblib.load(DEFAULT_MODEL_PATH)

    def predict(self, data: Union[List[Any], dict[str, Any], pd.DataFrame]) -> List[bool]:
        # data — это либо список значений в порядке FEATURES или словарь с нужными ключами
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:  # список значений
            df = pd.DataFrame([data], columns=FEATURES)
        df = df[FEATURES]
        preds = self._pipeline.predict(df)
        return [bool(x) for x in preds]

    __call__ = predict

    def __repr__(self) -> str:
        return f"<HeartAttackModel path='{DEFAULT_MODEL_PATH.name}'>"
