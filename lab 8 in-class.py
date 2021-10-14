#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:39:38 2021

Predict interest rates on loans from LendingTree

@author: leespitzley
"""

#%% imports

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import metrics

#%% load data

ld = pd.read_csv('data/lendingclub_2015-2018.csv')
ld.head()


#%% check interest rates

ld['int_rate'].hist()


#%% clean loan duration

# view unique values
ld['term'].unique()

# split rows into parts
term_split = ld['term'].str.split(' ')

# view first five rows
print(term_split[:5])

#%% get duration from split string
# the str function can retrieve a specific list element for all rows
term_split.str[1]
ld['duration'] = term_split.str[1]

# add this to the dataframe
print(ld['duration'].head())
# this column is not in integer format. Must fix!

#%% convert column to integer

# convert column to integer
ld['duration'] = ld['duration'].apply(int)
print(ld['duration'].head())

#%% rescale income & loan amount

ld['log_loan_amnt'] = np.log(ld['loan_amnt'])
ld['log_annual_inc'] = np.log(ld['annual_inc'] + 1) # log must be >0, otherwise it gives a warning


np.log(1)
np.log(100)
np.log(10000)

#%% get correlations
cols = ['int_rate', 'log_loan_amnt', 'installment', 'log_annual_inc', 'duration', 'fico_range_low', 'revol_util', 'dti']
corr = ld[cols].corr()
# format it with colors for Markdown reports
corr.style.background_gradient(cmap='coolwarm')

#%% specify our predictor variables

pred_vars = ['log_loan_amnt', 'log_annual_inc', 'fico_range_low', 'revol_util', 'dti', 'duration']

#%% drop rows with missing data
print("before dropping rows with missing data", len(ld))
ld = ld.dropna(subset=pred_vars)
print("after dropping rows with missing data", len(ld))

#%% train test split
from sklearn.model_selection import train_test_split

# use index-based sampling since we have time series data
train, test = train_test_split(ld, test_size=0.25, shuffle=False)

#%% view dates in the dataset

# earliest and latest dates in train
print("training data starts\n", train['issue_d'].head())
print("training data ends\n", train['issue_d'].tail())
# earliest and latest in test
print("testing data starts\n", test['issue_d'].head())
print("testing data ends\n", test['issue_d'].tail())

#%% fit simple regression with credit score only (FICO)

reg_fico = sm.OLS(train['int_rate'], train['fico_range_low']).fit()
print(reg_fico.summary())


#%% regression with multiple predictors

reg_multi = sm.OLS(train['int_rate'], train[pred_vars], hasconst=False).fit()
reg_multi.summary()

#%% evaluation

print(reg_fico.aic)
print(reg_multi.aic)


sm.stats.anova_lm(reg_fico, reg_multi)


#%% RMSE for the models

fico_pred = reg_fico.predict(test['fico_range_low'])

fico_rmse = metrics.mean_squared_error(test['int_rate'], fico_pred, squared=False)
print("RMSE:", fico_rmse)


multi_pred = reg_multi.predict(test[pred_vars])

multi_rmse = metrics.mean_squared_error(test['int_rate'], multi_pred, squared=False)
print("RMSE:", multi_rmse)

