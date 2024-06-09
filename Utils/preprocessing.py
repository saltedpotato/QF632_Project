import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 632

def get_standard_train_test_split(df, test_ratio=0.2, is_time_series = False):
    full_data = df.copy() # make a copy to ensure same df is not edited twice over
    X = full_data.drop(columns="y")
    y = full_data["y"]

    # Do not shuffle train test split if data is time series to avoid "seeing into future"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=RANDOM_STATE, shuffle = is_time_series)

    return X, y, X_train, X_test, y_train, y_test

def Eskin(data):
  r=data.shape[0]
  s=data.shape[1]
  num_cat=data.nunique().values
  agreement=np.zeros(s)
  eskin= np.zeros(shape=(r,r))
  for i in range(r-1):
    for j in range(1+i, r):
      for k in range(s):
        if data.iat[i, k] == data.iat[j, k]:
          agreement[k] = 1
        else:
          agreement[k] = num_cat[k]**2/(num_cat[k]**2 + 2)
      eskin[i][j] = (s/sum(agreement))-1
      eskin[j][i] = eskin[i][j]
  return eskin

