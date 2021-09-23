#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:01:07 2021

Lab 5 in-class

@author: leespitzley
"""

#%% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#%% data

hf = pd.read_csv('data/creditcard.csv')

#%% basic stats

display(hf.describe())

#%% train/test split

np.random.seed(516)

# create train and test
train, test = train_test_split(hf, test_size=0.25)
print("Rows in train:", len(train))
print("Rows in test:", len(test))

# view some stats by different variables
train_stats = train.groupby('Class')[['Time', 'Amount', 'V1']].agg(['mean', 'count'])
print("Training data:\n", train_stats)
test_stats = test.groupby('Class')[['Time', 'Amount', 'V1']].agg(['mean', 'count'])
print("Testing data:\n", test_stats)

#%% predictor variables

pred_vars = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
             'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
             'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
             'V26', 'V27', 'V28'] 

#%% create a decision tree

dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
dtree.fit(train[pred_vars], train['Class'])

#%% random forest
rf = ensemble.RandomForestClassifier(max_depth=10)
rf.fit(train[pred_vars], train['Class'])

#%% neural network

mlp = MLPClassifier(hidden_layer_sizes=(20,20,10))
mlp.fit(train[pred_vars], train['Class'])

#%% SVM

svc = svm.SVC(probability=True)
svc.fit(train[pred_vars], train['Class'])

#%% Naive Bayes

nb = GaussianNB()
nb.fit(train[pred_vars], train['Class'])

#%% Logistic Regression

lr = LogisticRegression()
lr.fit(train[pred_vars], train['Class'])


#%% evaluation

# list of our models
fitted = [dtree, rf, mlp, svc, nb, lr]

# empty dataframe to store the results
result_table = pd.DataFrame(columns=['classifier_name', 'fpr','tpr','auc', 
                                     'log_loss', 'clf_report'])

for clf in fitted:
    # print the name of the classifier
    print(clf.__class__.__name__)
    
    # get predictions
    yproba = clf.predict_proba(test[pred_vars])
    yclass = clf.predict(test[pred_vars])
    
    # auc information
    fpr, tpr, _ = metrics.roc_curve(test['Class'],  yproba[:,1])
    auc = metrics.roc_auc_score(test['Class'], yproba[:,1])
    
    # log loss
    log_loss = metrics.log_loss(test['Class'], yproba[:,1])
    
    # add some other stats based on confusion matrix
    clf_report = metrics.classification_report(test['Class'], yclass)
    
    # add the results to the dataframe
    result_table = result_table.append({'classifier_name':clf.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc,
                                        'log_loss': log_loss,
                                        'clf_report': clf_report}, ignore_index=True)
    
#%% show table

result_table.set_index('classifier_name', inplace=True)
print(result_table)

#%%  print classification

for i in result_table.index:
    print('\n---- statistics for', i, "----\n")
    print(result_table.loc[i, 'clf_report'])
    print("Model log loss:", result_table.loc[i, 'log_loss'])
    
#%% plot ROC curve

fig = plt.figure(figsize=(14,12))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()

#%% precison recall curve

for clf in fitted:
    metrics.plot_precision_recall_curve(clf, test.loc[:, pred_vars], test['Class'])