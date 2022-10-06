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
 incorrect^^^    ^^^correct predictions
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

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

'''

precision_score(y_train_5, y_train_pred) - 3530 / (3530 + 687)
Out[3]: 0.8370879772350012
- model is correct 83.7% of times, when it predicts a five

recall_score(y_train_5, y_train_pred) - 3530 / (3530 + 1891)
Out[4]: 0.6511713705958311
- model detects 65.1% of fives

'''

# %%%% F1 score

'''

Precision and recall may be combined into an F1 score


F1 = TP / (
    TP + ((FN + FP) / 2))

F1 = precision * recall / (precision + recall)

'''

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

'''

Out[5]: 0.7325171197343846

F1 score favors classifiers with similar precision and recall

'''

# %%%% Precision/recall trade-off

'''

For a classifier of videos suitable for kids it's better that a model has 
low recall (rejects many good videos) and high precision (keeps ONLY safe ones)

For a classifier of shoplifters on a camera footage it's better with
low precision (like 30%, signaling lots of non-shoplifters) and high recall
(many false alerts, but still better - guards will decide).


Stochastic Gradient Descent, for example, calculates the score for each
object based on decision function. If the score is greater than threshold,
then classifier assigns positive class, if less - negative.

It is possible to display decision scores in scikit learn.

'''

some_digit = X.iloc[0,]

some_digit_image = np.array(some_digit).reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")

y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

'''

y_some_digit_pred
Out[21]: array([ True])

'''


threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

'''

y_some_digit_pred
Out[25]: array([False])

Threshold increased and another class (FN) was assigned to a five.
Raising threshold decreases recall.

How to define threshold correctly?

'''

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
precision_score(y_train_5, y_train_pred_90)
# Out[35]: 0.9000345901072293

recall_score(y_train_5, y_train_pred_90)
# Out[36]: 0.4799852425751706

# %%%% ROC Curve

'''

Receiver operating characteristic curve. Plots true positive rate (TPR
or recall) against false positive rate (FPR) — ratio of negative instances
incorrectly classified as positive. 

'''
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    
plot_roc_curve(fpr, tpr)
plt.show()

# TODO Finish plots

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
# Out[42]: 0.9604938554008616

'''

A rule of a thumb is to use PR curve when positive class is rare or when you
care more about the false positives than false negatives. Otherwise use ROC.

In this example ROC gives very high score, but this is due to relatively
small number of positive vs negative instances in the dataset.

'''

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
'''

In Random Forest there's "predict proba" instead of "decision_function"
in SGDClassifier. Predict proba gives a row per instance and
a column per class - array of probabilities
(that given instance belongs to given class)

Use positive class probability as a score

'''

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,
                                                      y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)
# Out[47]: 0.9983436731328145

# %% Multiclass Classification
# Or multinomial classifiers

'''

It's possible to train 10 binary classifiers to classify numbers
into ten classes — One-versus-rest (OvR) strategy (or one vs all).
Decision is made based on choosing a classifier with the highest score.

OvO - one-versus-one strategy is to train binary classifiers
on pairs of numbers "0 or 1" etc.
With N classes, N * (N - 1) / 2 classifiers will be needed.

OvO is preferred for algorithms which scale poorlyto large datasets like SVM
For other binary classifiers usually OvR is preferred.

'''

from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])

# Out[48]: array([5], dtype=int8) - Correct prediction

some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores

'''
array([[ 1.72501977,  2.72809088,  7.2510018 ,  8.3076379 , -0.31087254,
         9.3132482 ,  1.70975103,  2.76765202,  6.23049537,  4.84771048]])
         ^^^^^^^^^
Scikit learn automatically uses OvO approach, and the score for 5 wins.

Didn't get it, there should be 45 classifiers in OvO right?

'''

np.argmax(some_digit_scores)
# Out[50]: 5

svm_clf.classes_
# Out[51]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int8)

from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC()) # fit very slow
ovr_clf.fit(X_train, y_train)

ovr_clf.predict([some_digit])
# Out[54]: array([5], dtype=int8)

# Train a multiple SGD classifier
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
# Out[57]: array([3], dtype=int8) - wrong!

sgd_clf.decision_function([some_digit])

'''

Out[58]: 
array([[-31893.03095419, -34419.69069632,  -9530.63950739,
          1823.73154031, -22320.14822878,  -1385.80478895,
        -26188.91070951, -16147.51323997,  -4604.35491274,
        -12050.767298  ]])

'''

cross_val_score(sgd_clf, X_train, y_train,
                cv=3, scoring="accuracy")
# Out[59]: array([0.87365, 0.85835, 0.8689 ])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train,
                cv=3, scoring="accuracy")
# Out[64]: array([0.8983, 0.891 , 0.9018])


# Train a multiple SGD classifier now on scaled data
# Also trying multithreading

from joblib import parallel_backend

with parallel_backend('threading', n_jobs=8):
    sgd_clf.fit(X_train_scaled, y_train)

sgd_clf.predict([some_digit])

# Out[68]: array([3], dtype=int8) - still wrong

