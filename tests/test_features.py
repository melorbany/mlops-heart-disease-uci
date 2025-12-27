import pandas as pd
from src.features.build_features import build_preprocessor
from src.config import NUMERIC_FEATURES, TARGET_COLUMN


def test_build_preprocessor():
    preprocessor = build_preprocessor()

    # Build dummy data
    df = pd.DataFrame(
        {
            "age": [63, 45],
            "trestbps": [145, 130],
            "chol": [233, 250],
            "thalach": [150, 140],
            "oldpeak": [2.3, 1.4],
            "sex": [1, 0],
            "cp": [3, 2],
            "fbs": [1, 0],
            "restecg": [0, 1],
            "exang": [0, 1],
            "slope": [0, 1],
            "ca": [0, 2],
            "thal": [1, 2],
            TARGET_COLUMN: [1, 0],
        }
    )

    X = df.drop(columns=[TARGET_COLUMN])
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    # At least same number of rows
    assert X_trans.shape[0] == X.shape[0]
    # Some features are created
    assert X_trans.shape[1] > len(NUMERIC_FEATURES)
