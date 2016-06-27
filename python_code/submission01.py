
# coding: utf-8

# # First submission
# Evaluation metric: Accuracy

# In[270]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ## DATA MUNGING

# In[300]:

def Munge(data):
    df = data.copy()

    # lower case the column names
    df.columns = df.columns.str.lower()

    # create dummies for sex variable
    df = df.join(pd.get_dummies(df.sex, prefix='sex'))
    df = df.join(pd.get_dummies(df.pclass, prefix='pclass'))
    df = df.join(pd.get_dummies(df.embarked, prefix='embarked'))

    # missing values for fares
    df.ix[df.fare.isnull(), 'fare'] = 0

    # impute missing ages as the average
    df.ix[df.age.isnull(),'age'] = df.age.mean()

    return df


# In[301]:

# load data
df_train = pd.read_csv('data/train.csv', index_col=0)
df_test = pd.read_csv('data/test.csv', index_col=0)

# munge data
df_train = Munge(df_train)
df_test = Munge(df_test)


# ## EDA
# - Females were more likely to survive
# - People from class 1 were more likely to survive

pd.crosstab(df_train.sex, df_train.survived, dropna=False, normalize='index')

pd.crosstab(df_train.pclass, df_train.survived, dropna=False, normalize='index')

pd.crosstab(df_train.embarked, df_train.survived, dropna=False, normalize='index')


# ## BASE MODELS

# train test split
train, val = train_test_split(df_train, test_size=0.3, random_state=0)

# features to exclude
excluded_features = ['survived','cabin','sex','ticket','name','embarked','pclass']
features = train.ix[:,~train.columns.isin(excluded_features)].columns
features


# ### Training set

train_x = train.ix[:,features]
train_y = train.survived

# logistic regression
lr = LogisticRegression()
lr.fit(train_x, train_y)
lr_pred_train = lr.predict(train_x)
print('lr train accuracy: {result}'.format(result=accuracy_score(train_y, lr_pred_train)))

# random forest
rf = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=0, n_jobs=3)
rf.fit(train_x, train_y)
rf_pred_train = rf.predict(train_x)
print('rf train accuracy: {result}'.format(result=accuracy_score(train_y, rf_pred_train)))


# ### Validation set

val_x = val.ix[:,features]
val_y = val.survived

# logistic regression
lr_pred_val = lr.predict(val_x)
print('lr validation accuracy: {result}'.format(result=accuracy_score(val_y, lr_pred_val)))

# random forest
rf_pred_val = rf.predict(val_x)
print('rf validation accuracy: {result}'.format(result=accuracy_score(val_y, rf_pred_val)))


# ### Initial submission

def SubmitCSV(data, filename):
    data = pd.Series(submission, index=df_test.index, name='Survived')
    pd.DataFrame(data).to_csv(filename)

submission = rf.predict(df_test.ix[:,features])
SubmitCSV(submission, 'submissions/randomforest02.csv')

