import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import xgboost as xgb
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Data/imalanced_data.csv')

#some predefined options for pandas
with open('pandasOptions.env', 'r') as f:
    for line in f:
        pd.set_option(line.split("=")[0], int(line.split("=")[1]))

#Some EDA
data.shape
data.columns
data.info()
data.head()


#drop some columns that won't be needed
drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
data.drop(drop_cols, axis=1, inplace=True)

data['type'].value_counts()
# CASH_OUT    2237500
# PAYMENT     2151495
# CASH_IN     1399284
# TRANSFER     532909
# DEBIT         41432

data['isFraud'].value_counts()
# 0    6354407
# 1       8213

#Data Preparation, create one hot encoder for categorical columns
data = pd.get_dummies(data, drop_first=True)
data = shuffle(data, random_state=20)

#define stratify function which will take the size of the smaller case
def stratify_data(data, dep_col):
    # define custom stratified samples
    fraud_data = data.loc[data[dep_col] == 1]
    non_fraud_data = data.loc[data[dep_col] == 0]
    if fraud_data.shape[0] < non_fraud_data.shape[0]:
        data_c = non_fraud_data.sample(n=fraud_data.shape[0])
    else:
        data_c = fraud_data.sample(n=non_fraud_data.shape[0])
    new_data = pd.concat([data_c, fraud_data], axis=0)
    new_data = shuffle(new_data)
    return new_data

#define list scores
list_scores = []
for i in range(10):
    print(f"[INFO] Stratified Iteration number: {i}")
    temp = stratify_data(data)
    X, y = temp.drop('isFraud', axis=1), temp['isFraud']
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.8)
    model = xgb.XGBClassifier()
    model.fit(train_X, train_Y)
    list_scores.append(model.score(test_X, test_Y))

np.mean(list_scores) #0.9891789057149379 score, which without statified sampling would be hard to achieve