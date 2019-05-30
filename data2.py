#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares data to analyze for unfair discrimination.

Created on Thu Apr 11 08:32:40 2019
@author: justin
"""
#%%  Parameters
# Include discriminatory features?
include_unfair_features = True

# Use a subset for a first pass?
use_small_sample = False

# Determine the threshold for including feature.
accuracy_threshold = 0

# Determine how many digits to round to in outputs.
rounding = 3


#%%  Head
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import classification_report, accuracy_score

print('Import ...')
t = time.time()
t1 = time.time()

X = pd.read_csv('train_values.csv', index_col = 'row_id')
y = pd.read_csv('train_labels.csv', index_col = 'row_id')
df = X.join(y)

# small df
if use_small_sample:
    df = df.iloc[:5000]

#X2 = pd.read_csv('test_values.csv', index_col = 'row_id')
X2 = False

round(time.time() - t)

#%%  Columns 
# Columns by subject
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

# columns by type
numeric_columns = census_cols + ['loan_amount', 'applicant_income']

# Columns with many categories
big_categorical_columns = property_cols + ['lender']

# Columns with few categories
small_categorical_columns = applicant_cols + loan_cols
remove_list = ['loan_amount', 'applicant_income', 'lender']
for item in remove_list:
    small_categorical_columns.remove(item)

len(X.columns) == len(numeric_columns) + len(small_categorical_columns) + len(big_categorical_columns)

# Columns to check discrimination
unfair_columns = ['applicant_ethnicity', 'applicant_race', 'applicant_sex']
ignore_columns = property_cols + ['minority_population_pct']

fair_columns = list(X.columns)
for column in X.columns:
    if column in unfair_columns or column in ignore_columns:
        fair_columns.remove(column)

# Which columns to examine
if include_unfair_features:
    print('Including unfair.')
    active_columns = unfair_columns + fair_columns
else:
    print('Including only fair.')
    active_columns = fair_columns 


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
#    X2[column] = X2[column].map(replace)

# Check - show minimum size of category.
df.lender.value_counts()
df.county_code.value_counts().iloc[-1]

#quick
round(time.time() - t)

#%%  Bin big categoricals
print('Bin big categoricals ...')
t = time.time()
df.groupby('county_code').accepted.mean()
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
#    X2[column] = X2[column].map(assign)

df.state_code.value_counts()
#X2.state_code.value_counts()

categorical_columns = small_categorical_columns + big_categorical_columns
round(time.time() - t)

#%%  Cleaning
df.shape
df = df.dropna()
df.shape

y = df.accepted
y.shape


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
    acc = accuracy_score(y, y2).round(rounding)
    
    print(column, acc)
    return acc

##  Features NP array
#%%  Encode features
print('\nEncode features ...') 
t = time.time()
Features = np.empty((df.shape[0],0))
   
for column in active_columns:
    x = df[column]
    if column in categorical_columns:
        x = encode_string(df[column])
    elif column in numeric_columns:
        x = np.array(df[column]).reshape(-1,1)
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
    else:
        raise ValueError('Untyped column')
    Features = np.concatenate([Features, x], axis = 1)
#    print('\n',column,x.shape[1])

round(time.time() -t)

#%%  Split
print('Split ...')
X, val_X, y, val_y = train_test_split(Features, y)
y.shape


#%%  Message
t2 = time.time()
print('Data Prepped.')
print('Time (min):', round((t2-t1)/60, 1) )
print('\a')