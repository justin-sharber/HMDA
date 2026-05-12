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
rnd = 3

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

fp = 'data.csv'
df = pd.read_csv(fp, index_col = 'row_id')
#df = pd.read_csv('big_data.csv', index_col = 'row_id')
df.shape

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

feature_columns = property_cols + loan_cols + applicant_cols + census_cols
X = df[feature_columns]
y = df.accepted

len(df.columns) == len(X.columns) + 1

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

round(time.time() - t)

#%%  Bin big categoricals
print('Bin big categoricals ...')
t = time.time()
big_categorical_columns

for column in big_categorical_columns:
    s = df.groupby(column).accepted.mean()//.1*10
    s = dict(s)
    def assign(val):
        if val in s.keys():
            return s[val]
        else:
            return 0.5
    df[column] = df[column].map(assign)

categorical_columns = small_categorical_columns + big_categorical_columns
round(time.time() - t)

#%%  Cleaning
df.shape
df = df.dropna()
df.shape

y = df.accepted
y.shape

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
    acc = accuracy_score(y, y2)
    acc = round(acc, rnd)
    print(column, acc)
    return acc

#%%  Test and include categoricals -------------------------------
print('\nCategoricals ...')    
Features = np.empty((df.shape[0],0))
column_accuracies = {}
good_columns = []

for column in categorical_columns:
    x = encode_string(df[column])
    acc = logistic_reg(column, x)
    column_accuracies.update({column: acc})
    
    if acc >= accuracy_threshold:
        good_columns.append(column)
        Features = np.concatenate([Features, x], axis = 1)

good_columns

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

#%%  Sort column accuracies
column_accuracies
sc = []
sa = []
vals = sorted(set(column_accuracies.values()))

for accuracy in vals:
    for column in column_accuracies.keys():
        if column_accuracies[column] == accuracy:
            sc.append(column)
            sa.append(accuracy)
                       
pd.Series(sa, index = sc)            
        
#%%  Split
print('\nSplit ...')
X, val_X, y, val_y = train_test_split(Features, y)
y.shape

#%%  Complete
t2 = time.time()
print('Data Prepped.', round((t2-t1)/60, 1), 'minutes. \a')