#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:18:15 2022

@author: artemii
"""

# %% Import MNIST Dataset

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

some_digit = X.iloc[2,]
some_digit_image = np.array(some_digit).reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
y[2]

y = y.astype(np.int8)
y[2]


# %% Split data

X_train, X_test, y_train, y_test = np.array(X[:60000]), np.array(X[60000:]), np.array(y[:60000]), np.array(y[60000:])


# %% Binary classifier
# This will only classify 5 or NOT 5

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

y_train_5.astype(np.int8)
# %%% Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])


# %% Evaluation

'''

Evaluating of a classifier is trickier than evaluating a regressor.

'''

# %%% Cross-validation

# %%%% Customization
# This explains how cross_val_score works

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
'''

0.9669
0.91625
0.96785

'''
# %%%% Sklearn way

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

'''

This seems to work much slower than a for loop

Also I'm stillconfused with how all the predictions differ with what
is shown in the book although using the seed.

Out[106]: array([0.87365, 0.85835, 0.8689 ])

'''

# %%%% Skewed data

'''

The code below demonstrates that in a highly skewed dataset if you always
predict one of the categories, you get very high accuracy — i.e. about 10%
of the dataset is represented by fives.
Hence the accuracy of "Never5Classifier"

'''
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        # Does nothing during training
        pass
    def predict(self, X):
        # returns a zero array of the X length
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

'''

Out[107]: array([0.91125, 0.90855, 0.90915])

"accuracy" is not the best performance metric,
especially for the skewed datasets

'''

# %%%% Confusion matrix

'''

This problem may be addressed by using a confusion matrix.
It computes how often a model confuses each class with another class.

'''

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

'''

This is similar to cross_val_scores, but returns predictions
of a model from cross-validation evaluation

'''

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

'''

Out[118]: 
array([[53892,   687],   <<< non-fives (negative class)
       [ 1891,  3530]])  <<< fives (positive class)
   correct^^^    ^^^incorrect predictions
Each row is an actual class, and each column is a predicted class
[1, 1] are true negatives, [2, 1] are false positives
[1, 2] are false negatives, [2, 2] are true negatives

A perfect confusion matrix looks like this

'''

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

'''

array([[54579,     0],
       [    0,  5421]])
A perfect prediction matrix has non-zero values only on the main diagonal.

'''

# %%%% Precision & Recall

'''

Precision is a more concise metric based on confusion matrix.
It is calculated as follows:
    TP / (TP + FP)
(where TP is true positive etc)

Recall
or Sensitivity or True Positive Rate (TPR) is used together with precision:
    TP / (TP + FN)
    

'''