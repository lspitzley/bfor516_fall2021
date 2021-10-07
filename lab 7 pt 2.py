#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:02:34 2021

@author: leespitzley
"""


#%% imports

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#%% read data
weather = pd.read_csv('data/last_ten_alb.csv')


weather['YEARMODA'] = pd.to_datetime(weather['YEARMODA'])
print("Most recent date in data:", weather['YEARMODA'].max())

#%% create lagged columns

most_recent_data = weather['YEARMODA'].max()
# https://stackoverflow.com/questions/39964558/pandas-max-value-index
most_recent_yday = weather.loc[weather['YEARMODA'].idxmax(), 'YDAY']
print("Most recent date in data:", weather['YEARMODA'].max())


# figure out starting dates for predictions
# just get one day into the future 
prediction_start = most_recent_data + timedelta(days=1)
prediction_start_yday = list(most_recent_yday + range(1,4))

# generate list of dates for predictions
eval_dates = pd.date_range(prediction_start, periods=3).tolist()


#%% add future dates to the dataframe

predictions_df = pd.DataFrame(eval_dates, columns=['YEARMODA'])
predictions_df['YDAY'] = prediction_start_yday


# add these to the main dataframe
weather = weather.append(predictions_df, ignore_index=True)
weather[['YEARMODA', 'YDAY', 'SLP', 'I_PRCP']].tail()


#%% create lagged columns

weather['SLP1'] = weather['SLP'].shift(1)
weather['SLP2'] = weather['SLP'].shift(2)
weather['SLP3'] = weather['SLP'].shift(3)

#%% just get the days we need to predict

prediction_set = weather[(weather['YEARMODA'] >= prediction_start)].copy()

prediction_set.head() # see the lagged columns on the end?

#%% bring in the random forest from last time
import pickle
file_name = 'rf_1day_prcp.sav'
rf = pickle.load(open(file_name, 'rb'))

#%% generate predictions
pred_vars = ['YDAY', 'SLP1']
prediction_set['P_PRCP'] = rf.predict_proba(prediction_set[pred_vars])[:,1] 
