#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:29:43 2021

Lab 2 in class notes


@author: leespitzley
"""

#%% imports

import numpy as np
import pandas as pd

#%% example showing directory
import os
# show working directory
os.getcwd()

# show files your working directory
os.listdir()

#%% format column names

names = pd.read_csv('data/kddcup.names', header=None, delimiter=':', skiprows=1)

# make column 0 into a list
name_list = names[0].tolist()

# add the last column with type
name_list.append('type')

print(name_list)

#%% read in full dataset

netattacks = pd.read_csv('data/kddcup.data_10_percent_corrected', names=name_list, header=None, index_col=None)


#%% view first five rows

netattacks.head()
netattacks.tail()


#%% basic descriptive statistics
netattacks.describe(include='all')

# store into a variable
stats_netattacks = netattacks.describe(include='all')

#%% store the descriptive stats

# store stats in a dataframe
df_stats = netattacks.describe(include='all')
# save dataframe to file
df_stats.to_csv('output/netattack_summary.csv')

#%% summarize by attack type
type_counts = netattacks.groupby('type').count()
type_means = netattacks.groupby('type').mean()


# get a multi-index with multiple stats
type_counts = netattacks.groupby('type').agg(['count', 'mean'])

# cleanest for just counts
type_counts = netattacks['type'].value_counts()
type_counts.head()


#%% plot histogram
netattacks['duration'].hist()


#%% count histogram
netattacks['count'].hist()

#%% correlations

netattacks['duration'].corr(netattacks['count'])

# correlations (all correlations)
corr_matrix = netattacks.corr()

# scatterplot
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html
netattacks.plot.scatter('duration', 'dst_host_diff_srv_rate')


#%% simplify to binary problem



# https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/
netattacks['label'] = np.where(netattacks['type'] == 'normal.', 'good', 'bad')
netattacks['label'].value_counts()
