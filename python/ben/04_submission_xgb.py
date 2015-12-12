
# coding: utf-8

from time import time
import datetime
from operator import itemgetter
import csv

import utils
import data_utils

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation as cv
from sklearn.grid_search import RandomizedSearchCV

import xgboost as xgb

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


print("Loading data sets")

train, test = data_utils.load_transformed_data()
X_train, y_train = data_utils.get_raw_values(train)


# Utility function to report best scores
def report(grid_scores, n_top=20):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(score.cv_validation_scores)
        print("Mean validation score: {0:.10f} (std: {1:.10f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


print("Starting XGB")

n_features = X_train.shape[1]
N_FOLDS = 10

params = {'colsample_bytree': 0.7810024008810753, 'silent': 0, 'subsample': 0.5455752693149476, 'seed': 42, 'objective': 'reg:linear', 'max_depth': 22}

model = xgb.XGBRegressor(**params)

# run randomized search
folds = cv.KFold(n=len(y_train), n_folds=N_FOLDS, shuffle=True, random_state=42)

scores = cv.cross_val_score(model, X_train, y_train, scoring=utils.rmspe_scorer, cv=folds, n_jobs=-1)
print(scores)
print(scores.mean())



