# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:40:28 2019

@author: lee
"""
import numpy as np
import pandas as pd
from matplotlib.pyplot import hist

#%% generate random data with numpy
matrix = np.random.randint(0,100,size=(100, 4))
print(matrix)
print(matrix[90:100,0])

#%%% make a dataframe with pandas
random_df = pd.DataFrame(matrix, columns=list('ABCD'))
random_df['A']
random_df['A'].plot.hist()

#%% generate from random normal
new_col = np.random.normal(loc=5, scale=2, size=100)
random_df['E'] = new_col
random_df['E'].plot.hist()

#%% generate new label column
labels = np.random.choice(['A_1', 'A_2', 'B_1', 'B_2'], size=100)
random_df['labels'] = labels
list(random_df)
label_group = random_df['labels'].str.split('_')
print(label_group)
random_df['group'] = label_group.str[0]
random_df.head()

#%% summarize the new dataframe by group
random_df.describe(include='all')
random_df.groupby('group')['A', 'B'].mean()
df_summary = random_df.groupby('group').mean()

#%% work on cars
mtcars = pd.read_csv('mtcars.csv')
list(mtcars) # lo
mtcars.describe()
mtcars.rename(columns={"Unnamed: 0": 'model'}, inplace=True)
list(mtcars)

#%% Lab
make = mtcars['model'].str.split(' ')
make.str[0]
mtcars['manufacturer'] = make.str[0]
mtcars.groupby('manufacturer')['mpg', 'hp'].mean()
mtcars['wt'].plot.hist()
mtcars['wt'].max()
