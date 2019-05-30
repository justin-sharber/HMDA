#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:22:05 2019

@author: justin
"""
#%%  Load Data
#from data2 import *

#%%  Packages
import numpy.random as nr
import time

from sklearn import preprocessing, linear_model
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

t1 = time.time()
nr.seed(0)

#%%  Logistic Model
rounding = 3

logistic_mod = linear_model.LogisticRegression(solver = 'lbfgs') 
logistic_mod.fit(X, y)
prediction = logistic_mod.predict(val_X)

acc = accuracy_score(val_y, prediction).round(rounding)
acc

#c1 = logistic_mod.coef_

#%%  End
t2 = time.time()
print('Time (min):', round((t2-t1)/60, 1) )
