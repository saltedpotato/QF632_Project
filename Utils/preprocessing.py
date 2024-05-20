import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 2024

def get_standard_train_test_split(df, test_ratio=0.2, is_time_series = False):
    full_data = df.copy() # make a copy to ensure same df is not edited twice over
    X = full_data.drop(columns="y")
    y = full_data["y"]

    # Do not shuffle train test split if data is time series to avoid "seeing into future"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=RANDOM_STATE, shuffle = is_time_series)

    return X, y, X_train, X_test, y_train, y_test