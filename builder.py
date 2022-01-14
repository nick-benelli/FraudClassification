#!/usr/bin/env python
# coding: utf-8

# # CPE 695 Final Project Team 1
# ## Neural Network
# https://archive.ics.uci.edu/ml/datasets/Audit+Data



# In[12]:


# Packages
import pandas as pd
import numpy as np
import warnings
from IPython.display import Image, HTML, display
import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# My Packages
import helper

# Load Data
from get_data import df

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


# In[13]:
# Variables
dep_col = 'Risk'
test_size = 0.2


# In[19]:


# Split
X = df.drop(columns = [dep_col])
y = df.loc[:, dep_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size)


# ## NN 1
# 
# Fit a neural network using independent variables ‘pclass + sex + age + sibsp’ and dependent variable ‘survived’. Fill in n/a attributes with the average of the same attributes from other training examples. Use 2 hidden layers and set the activation functions for both the hidden and output layer to be the sigmoid function. Set “solver” parameter as either SGD (stochastic gradient descend) or Adam (similar to SGD but optimized performance with mini batches). You can adjust parameter “alpha” for regularization (to control overfitting) and other parameters such as “learning rate” and “momentum” as needed.

# In[20]:


print()
model_name = 'MLP classifier'
print(model_name)
clf = MLPClassifier(
    solver='adam',
    hidden_layer_sizes=(2,),
    activation='logistic',
    learning_rate='constant',
    learning_rate_init=0.1,
    alpha=0.00000001, 
    momentum=0.4)

clf = clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)

print("Accuracy score of {} is: {}".format(model_name, round(accuracy, 4)))


# In[22]:


