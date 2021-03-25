import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
import xgboost as xgb

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

#Lets define a model without any concerns about the imbalanced data
X = data.drop(['isFraud'], axis=1)
y = data['isFraud']
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)

model_basic = xgb.XGBClassifier()
model_basic.fit(train_X, train_y)

pred_y = model_basic.predict(test_X)

print(f"Precision: {precision_score(test_y, pred_y)}\nF1 score: {f1_score(test_y, pred_y)}\nRecall: {recall_score(test_y, pred_y)} ")
# Precision: 0.9581831290555155
# F1 score: 0.8697643979057592
# Recall: 0.7962852007189934
# We can see that the he model without any stratified sampling is doing not that bad but we can still improve

#Lets try to add some weights to the model
ratio_scale_pos = round(data.loc[data['isFraud'] == 0].shape[0]/data.loc[data['isFraud'] == 1].shape[0])
model_weighted = xgb.XGBClassifier(scale_pos_weight=ratio_scale_pos)
model_weighted.fit(train_X, train_y)

pred_y = model_weighted.predict(test_X)

print(f"Precision: {precision_score(test_y, pred_y)}\nF1 score: {f1_score(test_y, pred_y)}\nRecall: {recall_score(test_y, pred_y)} ")
# Precision: 0.4268041237113402
# F1 score: 0.5960050386899406
# Recall: 0.9874776386404294
# We can see that the he model is better for infering the actual positives but poor on predicting the predicted positives
# with ratio_scale_pos = 774, but let's take the sqrt of the initial ratio_scale_pos, this will limit the effect of a multiplication
# of positive examples

ratio_scale_pos = round(np.sqrt(ratio_scale_pos)) #28
model_weighted_1 = xgb.XGBClassifier(scale_pos_weight=ratio_scale_pos)
model_weighted_1.fit(train_X, train_y)

pred_y = model_weighted_1.predict(test_X)

print(f"Precision: {precision_score(test_y, pred_y)}\nF1 score: {f1_score(test_y, pred_y)}\nRecall: {recall_score(test_y, pred_y)} ")
# Precision: 0.5956284153005464
# F1 score: 0.7394843962008141
# Recall: 0.9749552772808586
# We can see increase in precision, which is good for our model
# Let's try one more value
ratio_scale_pos = round(np.log(774))
model_weighted_2 = xgb.XGBClassifier(scale_pos_weight=ratio_scale_pos)
model_weighted_2.fit(train_X, train_y)

pred_y = model_weighted_2.predict(test_X)

print(f"Precision: {precision_score(test_y, pred_y)}\nF1 score: {f1_score(test_y, pred_y)}\nRecall: {recall_score(test_y, pred_y)} ")
# Precision: 0.7590243902439024
# F1 score: 0.834987925945801
# Recall: 0.9278473464519976
#Better that the previous one, there is still room for improvement using hyperparameter tunning

#Let's try the stratified approach:
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
list_precision = []
list_recall = []
list_f1 = []
for i in range(10):
    print(f"[INFO] Stratified Iteration number: {i}")
    temp = stratify_data(data)
    X, y = temp.drop('isFraud', axis=1), temp['isFraud']
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.8)
    model = xgb.XGBClassifier()
    model.fit(train_X, train_Y)
    pred_y = model.predict(test_X)
    list_f1.append(f1_score(pred_y, test_Y))
    list_recall.append(precision_score(pred_y, test_Y))
    list_precision.append(recall_score(pred_y, test_Y))

print(f"Precision: {np.mean(list_precision)}\nF1 Score: {np.mean(list_f1)}\nRecall: {np.mean(list_recall)}")
# Precision: 0.9843830119138065
# F1 Score: 0.9893688446704278
# Recall: 0.9944064824712339
#I think we achieved far better model with a few lines stratified sampling, still room for improvement using hyperparameter tuning
