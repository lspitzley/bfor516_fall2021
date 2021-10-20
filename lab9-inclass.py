#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:01:35 2021

Lab 9 in class work

@author: leespitzley
"""

#%% imports

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics

#%% read data

ld = pd.read_csv('data/lendingclub_2015-2018.csv')

#%% subsampling

ld = ld.sample(100000, random_state=516)

#%% descriptive stats

ld['loan_amnt'].hist()
ld['annual_inc'].hist()

#%% log transform income

ld['log_annual_inc'] = np.log(ld['annual_inc']+1)
ld['log_annual_inc'].hist()

#%% clean up loan duration

# view unique values
ld['term'].unique()

# split rows into parts
term_split = ld['term'].str.split(' ')

# view first five rows
print(term_split[:5])

# the str function can retrieve a specific list element for all rows
term_split.str[1]
ld['duration'] = term_split.str[1]

# add this to the dataframe
print(ld['duration'].head())
# this column is not in integer format. Must fix!

# convert column to integer
ld['duration'] = ld['duration'].apply(int)
print(ld['duration'].head())

#%% correlations

cols = ['int_rate', 'loan_amnt', 'installment', 'log_annual_inc', 'duration', 'fico_range_low', 'revol_util', 'dti']
corr = ld[cols].corr()
corr.style.background_gradient(cmap='coolwarm')

#%% predictor columns

pred_vars = ['loan_amnt', 'log_annual_inc', 'fico_range_low', 'revol_util', 'dti', 'duration']


#%% drop missing data
print("before dropping rows with missing data", len(ld))
ld = ld.dropna(subset=pred_vars)
print("after dropping rows with missing data", len(ld))

#%% train and test split
from sklearn.model_selection import train_test_split

# use index-based sampling since we have time series data
train, test = train_test_split(ld, test_size=0.25, shuffle=False)   

#%% train test split stats
# earliest and latest dates in train
print("training data starts\n", train['issue_d'].head())
print("training data ends\n", train['issue_d'].tail())
# earliest and latest in test
print("testing data starts\n", test['issue_d'].head())
print("testing data ends\n", test['issue_d'].tail())

#%% OLS regression
reg_multi = sm.OLS(train['int_rate'], train[pred_vars], hasconst=False).fit()
reg_multi.summary()


#%% random forest regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()

rf_reg.fit(train[pred_vars], train['int_rate'])

#%% SVR

from sklearn.svm import LinearSVR

svr_reg = LinearSVR(max_iter=10000)

svr_reg.fit(train[pred_vars], train['int_rate'])

#%% MLP

from sklearn.neural_network import MLPRegressor

mlp_reg = MLPRegressor()

mlp_reg.fit(train[pred_vars], train['int_rate'])

#%% evaluation

models = [reg_multi, rf_reg, svr_reg, mlp_reg]

for reg in models:
    
    reg_pred = reg.predict(test[pred_vars])

    reg_rmse = metrics.mean_squared_error(test['int_rate'], reg_pred, squared=False)
    print(reg, "RMSE:", reg_rmse)