#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:43:26 2021

@author: leespitzley
"""

#%% imports
# this will not run
import pandas as pd
import datetime

#%% import data

county = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

#%%

print(county.describe(include='all'))

#%% get date
today = datetime.date.today()
print(today)
output = 'output/capital_comparison_' + str(today) + '.jpg'
date_title = 'COVID-19 Cases in the Capital Region (' + str(today) + ")"

#%% see data types

print(county.dtypes)


# convert dates

county['date'] = pd.to_datetime(county['date'])

#%%
print(county.describe(include='all', datetime_is_numeric=True))


#%% lab start

# Set the names of the county
cr = ['Albany', 'Columbia', 'Fulton', 'Greene', 'Montgomery', 'Rensselaer', 
      'Saratoga', 'Schenectady', 'Schoharie', 'Warren', 'Washington']

# get the first list element

# select only the data from NY where the county names are in the list above
alb =  county[(county['state'] == 'New York') & (county['county'].isin(cr))] # & (county['cases'] > 0)]


#%% groupby

example = alb.groupby(['date'])['cases'].sum().plot(logy=True)


#%% plot cases by county

# create a plot of cases by county
county_plot = alb.groupby(['date', 'county'])['cases'].sum().unstack().plot(logy=True, figsize=(10,5))
county_plot.set(xlabel='Date', ylabel='Number of Cases', title=date_title)
county_plot.get_figure()

#%% save to file
# Save the figure
county_plot.get_figure().savefig(output, bbox_inches='tight', dpi=300)


#%% 

"""
This is a block comment

The only purpose is to show the change in Git.

"""