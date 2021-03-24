#weighted xgboost model
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, weights=[0.99], flip_y=0.2, random_state=420)

#summarize class distribution
counter = Counter(y)
print(counter)

#viz the data
for lbl, pnt in counter.items():
    print(lbl, pnt)
    plt.scatter(X[np.where(y == lbl)[0], 0], X[np.where(y == lbl)[0], 1], label=str(lbl))
plt.legend()

#Baseline performance
#define a model
model = XGBClassifier()

#define the repeated 10 fold cross validation with 3 repeats
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

#roc_auc score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
print(f"Mean ROC AUC is: {np.mean(scores)}")
# Mean ROC AUC is: 0.5337778425656653 --> pretty bad
