#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:28:29 2021

Week 7 in class


@author: leespitzley
"""

#%% imports

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#%% import data

weather = pd.read_csv('data/last_ten_alb.csv')


weather['YEARMODA'] = pd.to_datetime(weather['YEARMODA'])
print("Most recent date in data:", weather['YEARMODA'].max())

#%% create lagged columns

weather['SLP1'] = weather['SLP'].shift(1)
weather['SLP2'] = weather['SLP'].shift(2)
weather['SLP3'] = weather['SLP'].shift(3)

#%% show the columns of interest

weather[['YEARMODA', 'YDAY', 'SLP', 'SLP1', 'SLP2', 'SLP3']].head()

#%% drop missing values

pred_vars = ['YDAY', 'SLP1']

# make list of all variables required to generate a forecast
model_vars = pred_vars + ['I_PRCP']

weather.dropna(subset=model_vars, inplace=True)

#%% train test split

train, test = np.split(weather, [int(.67 *len(weather))])

#%% train the model

rf = RandomForestClassifier(max_depth=5)

rf.fit(train[pred_vars], train['I_PRCP'])

#%% evalute the model

# get predictions
pred_p_prcp = rf.predict_proba(test.loc[:, pred_vars])
pred_i_prcp = rf.predict(test.loc[:, pred_vars])

# evaluation

metrics.plot_confusion_matrix(rf, test.loc[:, pred_vars], test['I_PRCP'])
print(metrics.classification_report(test['I_PRCP'], pred_i_prcp, digits=5))

#%% save (pickle)

#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
import pickle
file_name = 'rf_1day_prcp.sav'
pickle.dump(rf, open(file_name, 'wb'))