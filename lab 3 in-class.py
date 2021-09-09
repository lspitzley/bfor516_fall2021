#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:31:08 2021

Lab 3 - Classification

@author: leespitzley
"""

#%% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import metrics

#%% read in the names

# load txt file
names = pd.read_csv('data/kddcup.names', header=None, delimiter=':',skiprows=1)

# make column 0 into a list
name_list = names[0].tolist()

# add the last column with type
name_list.append('type')


#%% read main data frame

netattacks = pd.read_csv('data/kddcup.data_10_percent_corrected', names=name_list, header=None, index_col=None)

#%% create labels

netattacks['label'] = np.where(netattacks['type'] == 'normal.', 'good', 'bad')
netattacks['label'].value_counts()


#%% train-test split

train, test = train_test_split(netattacks, test_size=0.25)
print("Rows in train:", len(train))
print("Rows in test:", len(test))

#%% fit the decision tree
# original:
# dt = tree.DecisionTreeClassifier()

# adjusted parameters
dt = tree.DecisionTreeClassifier(max_depth=4, criterion='entropy')

# train the model using a list of column names
pred_vars = ['src_bytes', 'dst_bytes', 'count']

# The value we are trying to predict is 'label'
dt.fit(train.loc[:, pred_vars], train['label'])

#%% predict labels for the test data

predicted = dt.predict(test.loc[:, pred_vars])
print(predicted[:5]) # show first five predictions

#%% get some basic info

from collections import Counter
# count test data
test_labels_stats = Counter(test['label'])
print("Labels in the test data:", test_labels_stats)

# count predicted
predicted_labels_stats = Counter(predicted)
print("Labels in the predictions:", predicted_labels_stats)


#%% confusion matrix

print(metrics.confusion_matrix(y_true=test['label'], y_pred=predicted, labels=['good', 'bad']))

metrics.plot_confusion_matrix(dt, test.loc[:, pred_vars], test['label'], labels=['good', 'bad'])
plt.show()


#%%% compute accuracy statistics

# compute baseline accuracy (predict all bad)
baseline = test_labels_stats['bad'] / (test_labels_stats['good'] + test_labels_stats['bad'])
print("Baseline accuracy is:", baseline)

# compute the observed accuracy
acc = metrics.accuracy_score(test['label'], predicted)
print("Observed accuracy is:", acc)


#%% full confusion matrix statistics

result = metrics.classification_report(test['label'], predicted, digits=4)
print(result)

#%% view tree
tree.plot_tree(dt)
plt.show()
