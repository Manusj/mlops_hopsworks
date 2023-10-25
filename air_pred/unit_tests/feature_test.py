import pytest
from air_pred.utils import data_preprocessing
import pandas as pd
import numpy as np

@pytest.mark.parametrize("date_str, expected_date",
        [
            ("2023-10-09 01:00", pd.Timestamp('2023-10-09 01:00:00')),
            ("2023-10-09 24:00", pd.Timestamp('2023-10-10 00:00:00')),
        ]
)
def test_convert_to_datetime(date_str,expected_date):
    assert data_preprocessing.convert_to_datetime(date_str) == expected_date


@pytest.mark.parametrize("input_df, nan_count",
        [
            (pd.DataFrame([[1, 10],
                           [11, 12],
                           [np.nan, np.nan]], columns=["date_time", "b"]), 0)
        ]
)
def test_date_time_feature(input_df,nan_count):
    print(data_preprocessing.clean_data(input_df))
    assert data_preprocessing.clean_data(input_df).isna().sum().sum() == nan_count
