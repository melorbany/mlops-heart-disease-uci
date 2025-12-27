
import pandas as pd

from src.data.preprocess import clean_data
from src.config import TARGET_COLUMN


def test_clean_data():
    # Create small dummy dataframe with missing values
    df = pd.DataFrame(
        {
            "age": [63, None, 45],
            "sex": [1, 0, None],
            "cp": [3, 2, 1],
            TARGET_COLUMN: [1, 0, 1],
        }
    )

    df_clean = clean_data(df)

    # No missing values
    assert df_clean.isna().sum().sum() == 0
    # Target column still present
    assert TARGET_COLUMN in df_clean.columns
    # Shape preserved
    assert df_clean.shape[0] == df.shape[0]
