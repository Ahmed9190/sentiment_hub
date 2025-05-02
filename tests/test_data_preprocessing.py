import os
import tempfile

import pandas as pd

from src.data_preprocessing import load_and_prepare_data


def test_load_and_prepare_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        df = pd.DataFrame(
            {
                "review": ["good", "bad", None],
                "sentiment": ["positive", "negative", "positive"],
            }
        )
        df.to_csv(temp_file_name, index=False)

        X, y = load_and_prepare_data(temp_file_name)

        assert len(X) == 2, "Expected 2 valid rows after dropping missing values"
        assert all(y.isin([0, 1])), "Target values should be binary (0 or 1)"
        assert len(X) == len(y), "Features and target should have the same length"

    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
