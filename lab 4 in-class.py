#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:57:17 2021


Lab 4 - Advanced Evaluation


@author: leespitzley
"""

#%% imports 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import metrics


#%% read in  the data

ccfraud = pd.read_csv('data/creditcard.csv')


#%% descriptive stats

ccfraud.describe()
# get the average and count for each type
ccstats = ccfraud.groupby('Class')['Amount'].agg(['mean', 'count'])
# stats for fraud by count and average transaction amount
print(ccstats)

#%%  get percent fraud transactions

# percent of fraudulent transactions
print("Fraudulent transaction ratio:", ccstats.loc[1, 'count']/ccstats['count'].sum())


#%% set random seed

np.random.seed(516)

#%% train test split

train, test = train_test_split(ccfraud, test_size=0.25)
print("Rows in train:", len(train))
print("Rows in test:", len(test))
train_stats = train.groupby('Class')['Amount'].agg(['mean', 'count'])
print("Training data:\n", train_stats)
test_stats = test.groupby('Class')['Amount'].agg(['mean', 'count'])
print("Testing data:\n", test_stats)


#%% view  columns

# view all columns
print(list(ccfraud.columns))

#%% get training columns

# use column names 
pred_vars = ['Time', 'Amount', 'V8', 'V1']
print(ccfraud.loc[:, pred_vars])

#%% train the model

dtree = tree.DecisionTreeClassifier(criterion="entropy")
dtree.fit(train.loc[:, pred_vars], train['Class'])

#%% get basic tree statistics
print(dtree.get_n_leaves())

print(dtree.get_depth())

#%% predict test data

pred_labels = dtree.predict(test.loc[:, pred_vars])
pred_labels[0:4]

#%% get confusion matrix

metrics.plot_confusion_matrix(dtree, test.loc[:, pred_vars], test['Class'])

#%% get classifiction report

print(metrics.classification_report(test['Class'], pred_labels, digits=5))


#%% probablistic predictions

pred_probs = dtree.predict_proba(test.loc[:, pred_vars])
pred_probs[0:5, :]

#%% get ROC/AUC

metrics.roc_auc_score(test['Class'], pred_probs[:,1])


#%% plot the curve
metrics.plot_roc_curve(dtree, test.loc[:, pred_vars], test['Class'])

#%% precision recall curve

metrics.average_precision_score(test['Class'], pred_probs[:,1])


#%% plot precision  recall curve

metrics.plot_precision_recall_curve(dtree, test.loc[:, pred_vars], test['Class'])


#%% log loss 

# this means nothing on its own, must
# be compared to log loss from other models
# that use the same data
print(metrics.log_loss(test['Class'], pred_probs[:,1]))


#%%