#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:21:34 2021

@author: leespitzley
"""
#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#%% load in the data
trial_data = pd.read_csv('data/trial_data.csv')

#%% count words with str.split()

# this way keeps punctuation (not desirable)
trial_data['words'] = trial_data['transcript'].str.split()
trial_data.loc[0]['words']

#%% better way

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
text = "This is my text. It icludes commas, question marks? and other stuff. Also U.S.."
tokens = tokenizer.tokenize(text)
print(tokens)

# comparison to previous method 

print(text.split())

#%% apply the tokenizer

trial_data['words'] = trial_data['transcript'].apply(tokenizer.tokenize)
print(trial_data['words'].head())

#%% count the words


trial_data['word_count_nltk'] = trial_data['words'].apply(len)
trial_data['word_count_nltk'].hist()

#%% VADER

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk_sentiment = SentimentIntensityAnalyzer()

#%% apply sentiment analysis

full_sent = trial_data['transcript'].apply(lambda x: nltk_sentiment.polarity_scores(x))
full_sent.head()

#%% get the compound score
trial_data['sentiment'] = full_sent.apply(lambda x: x['compound'])
print(trial_data['sentiment'].head())

#%% plot
trial_data['sentiment'].hist()

#%% predictive analysis


pred_vars = ['word_count_nltk', 'sentiment']

# define the dependent variable
outcome = 'condition'

#%% train test split


np.random.seed(516)

# create train and test
train, test = train_test_split(trial_data, test_size=0.20, stratify=trial_data[outcome])
print("Rows in train:", len(train))
print("Rows in test:", len(test))

#%% random forest
from sklearn.ensemble import RandomForestClassifier

params = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15, None]}

rf_tuned = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='roc_auc')
rf_tuned.fit(train[pred_vars], train[outcome])


#%% Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier

params = {'hidden_layer_sizes': [(100,), (10,10), (5,5,5)], 
          'solver': ['adam', 'lbfgs', 'sgd']}

nnet_tuned = GridSearchCV(MLPClassifier(), param_grid=params, scoring='roc_auc')
nnet_tuned.fit(train[pred_vars], train[outcome])

#%% adaboost
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier

params = {'n_estimators': [10, 25, 50]}

ada_tuned = GridSearchCV(AdaBoostClassifier(), param_grid=params, scoring='roc_auc')
ada_tuned.fit(train[pred_vars], train[outcome])


#%% 

from sklearn import metrics

fitted = [rf_tuned, nnet_tuned, ada_tuned]

result_table = pd.DataFrame(columns=['classifier_name', 'fpr','tpr','auc', 
                                     'log_loss', 'clf_report'])

for clf in fitted:
    print(clf.estimator)
    yproba = clf.predict_proba(test[pred_vars])
    yclass = clf.predict(test[pred_vars])
    
    # auc information
    """
    Note that I specified the positve case here as 'truth'
    since that is what we are trying to detect. Otherwise,
    this line will present an error, since the classes are not
    0 or 1, but categorical labels.
    """
    fpr, tpr, _ = metrics.roc_curve(test[outcome],  yproba[:,1], pos_label='truth')
    auc = metrics.roc_auc_score(test[outcome], yproba[:,1])
    
    # log loss
    log_loss = metrics.log_loss(test[outcome], yproba[:,1])
    
    # add some other stats based on confusion matrix
    clf_report = metrics.classification_report(test[outcome], yclass)
    
    
    result_table = result_table.append({'classifier_name':str(clf.estimator),
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc,
                                        'log_loss': log_loss,
                                        'clf_report': clf_report}, ignore_index=True)
    


result_table.set_index('classifier_name', inplace=True)
# print(result_table)



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


#%% confusion matrix stats
for i in result_table.index:
    print('\n---- statistics for', i, "----\n")
    print(result_table.loc[i, 'clf_report'])
    print("Model log loss:", result_table.loc[i, 'log_loss'])

