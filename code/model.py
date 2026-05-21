#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model and predict

Created on Thu Apr 11 08:23:38 2019
@author: justin
"""
#%%  Load
from data import *

#%%  Packages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#%%  Logistic Model
nr.seed(0)

logistic_mod = linear_model.LogisticRegression() 

logistic_mod.fit(X, y)

prediction = logistic_mod.predict(val_X)

print('Basic Logistic')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()

#%%  Tree
from sklearn.tree import DecisionTreeRegressor

nr.seed(0)
tree = DecisionTreeRegressor(random_state=1, max_depth = 12)
tree.fit(X, y)
prediction = tree.predict(val_X).round()

print('Tree')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()

#%%  Random Forest
from sklearn.ensemble import RandomForestClassifier

nr.seed(0)
rf_clf = RandomForestClassifier(n_estimators=40)
rf_clf.fit(X, y)
prediction = rf_clf.predict(val_X)

print('Forest')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()

print()

#%%  Neural Network
from sklearn.neural_network import MLPClassifier
t = time.time()

nr.seed(0)
nn_mod = MLPClassifier(hidden_layer_sizes = (30, 30), max_iter= 40)
nn_mod.fit(X, y)
prediction = nn_mod.predict(val_X)

print('Neural Network')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()
print()

round(time.time() - t)

#%%
from sklearn import svm

nr.seed(0)
svm_mod = svm.LinearSVC()
svm_mod.fit(X, y)
prediction = svm_mod.predict(val_X)

print('SVM')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()

print()

#%%  Adaboost
from sklearn.ensemble import AdaBoostClassifier

nr.seed(0)
ab_clf = AdaBoostClassifier()
ab_clf.fit(X, y)
prediction = ab_clf.predict(val_X)

print('AdaBoost')
acc=accuracy_score(val_y, prediction)
a2 = round(acc, rnd)
print('Accuarcy:', a2)
print()

#%%  End
print('\n - Done \a')
