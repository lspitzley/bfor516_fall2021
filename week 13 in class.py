#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:25:29 2021


Topics of presidential debates


@author: leespitzley
"""

#%% imports

import numpy as np
import pandas as pd
from nltk.sentiment import vader


#%% import data

debates = pd.read_csv('data/all_debates.csv')

print(debates.describe())


#%% text clean



from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora

debates['clean_text'] = debates['text'].apply(preprocess_string)
print(debates.loc[1, ['text', 'clean_text']])


#%% dictionary


dictionary = corpora.Dictionary(debates['clean_text'])
print(dictionary)


#%% bag of words
bow_corpus = [dictionary.doc2bow(text) for text in debates['clean_text']]

#%% tf-idf version
from gensim import models

tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]

lda_tfidf = models.LdaModel(tfidf_corpus, num_topics=50, id2word=dictionary)
for topic in lda_tfidf.show_topics(num_topics=50):
    print("Topic", topic[0], ":", topic[1])


#%% join the topics to the original dataframe



from gensim.matutils import corpus2csc
all_topics = lda_tfidf.get_document_topics(bow_corpus, minimum_probability=0.0)
all_topics_csr = corpus2csc(all_topics)
all_topics_numpy = all_topics_csr.T.toarray()
all_topics_df = pd.DataFrame(all_topics_numpy)

classification_df = pd.concat([debates, all_topics_df], axis=1)



#%% sentiment

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk_sentiment = SentimentIntensityAnalyzer()

#%% apply sentiment analysis

full_sent = classification_df['text'].apply(lambda x: nltk_sentiment.polarity_scores(x))
full_sent.head()

#%% get the compound score
classification_df['sentiment'] = full_sent.apply(lambda x: x['compound'])
print(classification_df['sentiment'].head())

#%% classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

# https://www.geeksforgeeks.org/range-to-a-list-in-python/
pred_vars = [*range(0,50)]
pred_vars.append('sentiment')

from sklearn.model_selection import train_test_split
train, test = train_test_split(classification_df, test_size=0.20, stratify=classification_df['party'])

rf.fit(train[pred_vars], train['party'])


#%% get confusion matrix
from sklearn import metrics

metrics.plot_confusion_matrix(rf, test.loc[:, pred_vars], test['party'])

#%% get classifiction report
pred_labels = rf.predict(test.loc[:, pred_vars])

print(metrics.classification_report(test['party'], pred_labels, digits=5))

