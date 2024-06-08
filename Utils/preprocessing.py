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

# This function takes in data before encoding
def Frequency_table(data):
  num_cat= []
  for col_num in range(len(data.columns)):
    col_name= data.columns[col_num]
    categories = list(data[col_name].unique())
    num_cat.append(len(categories))
  r = data.shape[0]
  s = data.shape[1]
  freq_table= np.zeros(shape=(max(num_cat),s))
  for i in range(s):
    for j in range(num_cat[i]):
      count= []
      for num in range(0, r):
        count.append(0)
      for k in range(0,r):
        if (data.iat[k,i] -1== j):
          count[k] = 1
        else:
          count[k] = 0
      freq_table[j][i] = sum(count)
  return freq_table

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

def Overlap(data):
  r = data.shape[0]
  s = data.shape[1]
  agreement= []
  for i in range(s):
    agreement.append(0)
  overlap= np.zeros(shape=(r,r))
  for i in range(r-1):
    for j in range(1+i, r):
      for k in range(s):
        if data.iat[i, k] == data.iat[j, k]:
          agreement[k] = 1
        else:
          agreement[k] = 0
        overlap[i][j] = 1-1/s*(sum(agreement))
        overlap[j][i] = overlap[i][j]
  return overlap

def IOF(data):
  data=data.astype('int')
  r = data.shape[0]
  s = data.shape[1]
  freq_table= Frequency_table(data)
  agreement= []
  for i in range(s):
    agreement.append(0)
  iof= np.zeros(shape=(r,r))
  for i in range(r-1):
    for j in range(1+i, r):
      for k in range(s):
        c = data.iat[i,k]-1
        d = data.iat[j,k]-1
        if (data.iat[i,k] == data.iat[j,k]):
          agreement[k] = 1
        else:
          if freq_table[c][k]==0 or freq_table[d][k]==0:
            agreement[k]=0
          else:
            agreement[k] = 1/(1+(np.log(freq_table[c][k])*np.log(freq_table[d][k])))

        iof[i][j] = (s/sum(agreement))-1
        iof[j][i] = iof[i][j]
  return iof

def OF(data):
  data=data.astype('int')
  r = data.shape[0]
  s = data.shape[1]
  freq_table= Frequency_table(data)
  agreement= []
  for i in range(s):
    agreement.append(0)
  of= np.zeros(shape=(r,r))
  for i in range(r-1):
    for j in range(1+i, r):
      for k in range(s):
        c = data.iat[i,k]-1
        d = data.iat[j,k]-1
        if (data.iat[i,k] == data.iat[j,k]):
          agreement[k] = 1
        else:
          if freq_table[c][k]==0 or freq_table[d][k]==0:
            agreement[k]=0
          else:
            agreement[k] = 1/(1+(np.log(freq_table[c][k])*np.log(freq_table[d][k])))
        of[i][j] = (s/sum(agreement))-1
        of[j][i] = of[i][j]
  return of

def Lin(data):
  data=data.astype('int')
  r = data.shape[0]
  s = data.shape[1]
  freq_table= Frequency_table(data)
  freq_rel= freq_table/r
  agreement= []
  for i in range(s):
    agreement.append(0)
  lin= np.zeros(shape=(r,r))
  weights= []
  for i in range(s):
    weights.append(0)
  for i in range(r-1):
    for j in range(1+i, r):
      for k in range(s):
        c = data.iat[i,k]-1
        d = data.iat[j,k]-1
        if (data.iat[i,k] == data.iat[j,k]):
          agreement[k] = 2* np.log(freq_rel[c][k])
        else:
          agreement[k] = 2* np.log(freq_rel[c][k] + freq_rel[d][k])
          weights[k]= np.log(freq_rel[c][k]) + np.log(freq_rel[d][k])
        if i == j:
          lin[i][j]= 0
        else:
          lin[i][j] = 1/(1/sum(weights)*(sum(agreement))) - 1
          lin[j][i] = lin[i][j]
  return lin