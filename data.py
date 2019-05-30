#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads data.

Created on Thu Apr 11 08:32:40 2019
@author: justin
"""
#%%  Parameters
# Determine the threshold for including feature - 0.55
accuracy_threshold = 0

# Determine how many digits to round accuracy.
rounding = 3

# Use a subset for a quick check?
use_small_sample = False

#%%  Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
from numpy.random import choice
import math
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report, accuracy_score

#%%  Data
print('Import ...')
t1 = time.time()
t = t1

X = pd.read_csv('train_values.csv', index_col = 'row_id')
y = pd.read_csv('train_labels.csv', index_col = 'row_id')
df = X.join(y)

X2 = pd.read_csv('test_values.csv', index_col = 'row_id')

if use_small_sample:
    df = df.loc[choice(df.index, 5000)]
    X2 = X2.loc[choice(X2.index, 1000)]

round(time.time() - t)

#%%  Columns by subject
'Property Columns'
property_cols = ['msa_md', 'state_code', 'county_code']

'Loan Info'
loan_cols = ['lender', 'loan_amount', 'loan_type', 'property_type',
                'loan_purpose', 'occupancy', 'preapproval']

'Applicant Info'
applicant_cols = ['applicant_income', 'applicant_ethnicity', 'applicant_race', 
        'applicant_sex', 'co_applicant']

'Census Info'
census_cols = ['population', 'minority_population_pct', 'ffiecmedian_family_income',
        'tract_to_msa_md_income_pct', 'number_of_owner-occupied_units',
        'number_of_1_to_4_family_units']

check = property_cols + loan_cols + applicant_cols + census_cols
len(check) == len(X.columns)

#%%  Columns by type
feature_columns = X.columns

numeric_columns = census_cols + ['loan_amount', 'applicant_income']

# Columns with many categories
big_categorical_columns = property_cols + ['lender']

# Columns with few categories
small_categorical_columns = applicant_cols + loan_cols
remove_list = ['loan_amount', 'applicant_income', 'lender']
for item in remove_list:
    small_categorical_columns.remove(item)

len(feature_columns) == len(numeric_columns) + len(small_categorical_columns) + len(big_categorical_columns)

#%%  Big categoricals
# Group small categories, clean unfamiliar cats from X2.
print('Group small categories ...')
t = time.time()

for column in ['county_code', 'lender']:
    # big - big enough to keep as a group
    big = df[column].value_counts()>50
    # dict of truth values for big groups.
    big = dict(big)
    def replace(group):
        if group in big.keys() and big[group]:
            return group
        else:
            return -1
    df[column] = df[column].map(replace)
    X2[column] = X2[column].map(replace)

round(time.time() - t)

#%%  Bin big categoricals
print('Bin big categoricals ...')
t = time.time()
big_categorical_columns

for column in big_categorical_columns:
#    print(column)
    s = df.groupby(column).accepted.mean()//.1*10
    s = dict(s)
    def assign(val):
        if val in s.keys():
            return s[val]
        else:
            return 0.5
    df[column] = df[column].map(assign)
    X2[column] = X2[column].map(assign)

categorical_columns = small_categorical_columns + big_categorical_columns
round(time.time() - t)

#%%  Cleaning
df.shape
df = df.dropna()
df.shape

y = df.accepted
y.shape

#assert False

#%%  Imputing (X2)
X2.isnull().values.any()

for column in feature_columns:
    nulls = X2[column].isnull()
    med = X2[column].median()
    X2.loc[nulls, column] = med

X2.isnull().values.any()

#%%  Duplicates
# None
# X.drop_duplicates(subset = 'customer_id', keep = 'first', inplace = True)

#%%  Imbalance
# Data is balanced
df.accepted.value_counts()


#%%  Functions
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    try:
        ohe = preprocessing.OneHotEncoder(categories = 'auto')
    except TypeError:
        ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

def logistic_reg(column, x):
    logistic_mod = linear_model.LogisticRegression(solver = 'lbfgs') 
    logistic_mod.fit(x, y)
    y2 = logistic_mod.predict(x)
    acc = accuracy_score(y, y2).round(2)
    
    print(column, acc)
    return acc

##  Features NP array
#%%  Test and include categoricals
# Fill in Features, F2, and good_columns
print('\nCategoricals ...')    
Features = np.empty((df.shape[0],0))
F2 = np.empty((X2.shape[0],0))
column_accuracies = {}
good_columns = []

for column in categorical_columns:
    x = encode_string(df[column])
    acc = logistic_reg(column, x)
    column_accuracies.update({column: acc})
    
    if acc >= accuracy_threshold:
        good_columns.append(column)
        Features = np.concatenate([Features, x], axis = 1)
        
        x2 = encode_string(X2[column])
        F2 = np.concatenate([F2, x2], axis = 1)

good_columns
Features.shape[1] == F2.shape[1]

#%%  Add numerics, Scale data
print('\nNumerics ...')

for column in numeric_columns:
    x = df[column]
    x = np.array(x).reshape(-1,1)
    acc = logistic_reg(column, x)
    column_accuracies.update({column: acc})     
    
    if acc >= accuracy_threshold:
        good_columns.append(column)
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        Features = np.concatenate([Features, x], axis = 1)
        
        x2 = X2[column] 
        x2 = np.array(x2).reshape(-1,1)
        x2 = scaler.transform(x2)
        F2 = np.concatenate([F2, x2], axis = 1)

#%%  Split
print('\nSplit ...')
X, val_X, y, val_y = train_test_split(Features, y)
y.shape

#%%  Complete
t2 = time.time()
print('Data Prepped.')
print('Time (min):', round((t2-t1)/60, 1) )
print('\a')
column_accuracies
