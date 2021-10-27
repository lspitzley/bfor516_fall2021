#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:13:20 2021

@author: leespitzley
"""
#%% imports
import numpy as np
import pandas as pd
from matplotlib.pyplot import hist


#%% generate random data with numpy
matrix = np.random.randint(0,100,size=(100, 4))
print(matrix)
print(matrix[90:100,0])


#%% convert to dataframe
random_df = pd.DataFrame(matrix, columns=list('ABCD'))
random_df['A']
random_df['A'].plot.hist()

#%% generate from random normal
new_col = np.random.normal(loc=5, scale=2, size=100)
random_df['E'] = new_col
random_df['E'].plot.hist()

#%% generate categorical column
labels = np.random.choice(['A_1', 'A_2', 'B_1', 'B_2'], size=100)
random_df['labels'] = labels


label_group = random_df['labels'].str.split('_')
print(label_group)
random_df['group'] = label_group.str[0]
random_df.head()
