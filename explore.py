#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore the data

Created on Thu Apr 11 08:33:26 2019
@author: justin
"""
#%%  Packages
import numpy.random as nr
from numpy.random import choice

import matplotlib.pyplot as plt

#%%  Data
from data import *

nr.seed(0)
c = choice(df.index, 5000)
dfa = df
dfa.applicant_income.describe()
 
#%%  Small categoricals  
# Want to replace with enumerate.    
fig, axarr = plt.subplots(3, 3, figsize = (12,10))
plt.subplots_adjust(hspace=.5)

for i in range(len(small_categorical_columns)):
    col = small_categorical_columns[i]
    data = dfa.groupby(col).accepted.mean()
    data.plot.bar(ax = axarr[i//3][i%3] )

#%%  Big categoricals plot
fig, axarr = plt.subplots(2,2, figsize = (12,8))
plt.subplots_adjust(hspace=.5)

for i in range(4):
    col = big_categorical_columns[i]
    data = dfa.groupby(col).accepted.mean()
    data.plot.bar(ax = axarr[i//2][i%2] )
    
#%%  Numerics
num2 = ['population', 'minority_population_pct', 'loan_amount', 
        'applicant_income', 'number_of_owner-occupied_units',]
sns.pairplot(dfa[num2]) 