#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:03:31 2021

@author: leespitzley
"""

#%% imports

import numpy as np
import pandas as pd


#%% read data

call_data = pd.read_csv('data/earningscall_fraud.csv')

print(call_data.describe())


# percent of fraud
print(call_data['Restatement Topic'].mean())

#%% clean the text

from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora

call_data['clean_text'] = call_data['Sentence'].apply(preprocess_string)
print(call_data.loc[1, ['Sentence', 'clean_text']])

#%% dictionary

dictionary = corpora.Dictionary(call_data['clean_text'])
print(dictionary)

#%% create bag of words

bow_corpus = [dictionary.doc2bow(text) for text in call_data['clean_text']]


#%% fit LDA

from gensim import models

lda_10 = models.LdaModel(bow_corpus, num_topics=10, id2word=dictionary)

#%% print the topics

for topic in lda_10.show_topics():
    print("Topic", topic[0], ":", topic[1])


#%% print the docs


for doc in bow_corpus[0:9]:
    print(lda_10.get_document_topics(doc))


#%% perplexity

print('Perplexity: ', lda_10.log_perplexity(bow_corpus))


#%% coherence

from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda_10, texts=call_data['clean_text'], dictionary=dictionary, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


#%% remove context-specific stopwords

stopwords = []
with open('data/stoplist.txt', 'r') as f:
    stopwords = f.read().splitlines()

#%% remove new stopwords

def remove_stopwords(text):
    """ preprocess string and remove words from custom stopword list. """
    result = []

    for word in preprocess_string(text):
        if word not in stopwords:
            result.append(word)
    return result

call_data['clean_newstop'] = call_data['Sentence'].apply(remove_stopwords)

#%% new  dictionary and bow


new_dictionary = corpora.Dictionary(call_data['clean_newstop'])
print(new_dictionary)

new_corpus = [new_dictionary.doc2bow(text) for text in call_data['clean_newstop']]

#%% fit new model

lda_new = models.LdaModel(new_corpus, num_topics=15, id2word=new_dictionary)

for topic in lda_new.show_topics(num_topics=15):
    print("Topic", topic[0], ":", topic[1])

#%% perplexity


print('Perplexity: ', lda_new.log_perplexity(new_corpus))

#%% coherence
coherence_model_lda = CoherenceModel(model=lda_new, texts=call_data['clean_newstop'], dictionary=new_dictionary, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#%% convert to dataframe

from gensim.matutils import corpus2csc
all_topics = lda_new.get_document_topics(new_corpus, minimum_probability=0.0) # get topics
all_topics_csr = corpus2csc(all_topics) # get topic probabilites for each
all_topics_numpy = all_topics_csr.T.toarray() # convert to array
all_topics_df = pd.DataFrame(all_topics_numpy) # convert to dataframe

classification_df = pd.concat([call_data, all_topics_df], axis=1)


#%% show descriptives  of new dataframe

classification_df.describe()

#%% use for classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate

n_splits = 5

pred_vars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]


scoring = ['accuracy', 'neg_log_loss', 'f1', 'roc_auc']
rf_base = RandomForestClassifier()
cv_rf = cross_validate(rf_base, classification_df[pred_vars], classification_df['Restatement Topic'], cv=StratifiedShuffleSplit(n_splits), scoring=scoring)
print(cv_rf)
print("Mean Accuracy:", cv_rf['test_accuracy'].mean())
print("Mean F1:", cv_rf['test_f1'].mean())
print("Mean ROC:", cv_rf['test_roc_auc'].mean())
print("Mean Log Loss:", cv_rf['test_neg_log_loss'].mean())