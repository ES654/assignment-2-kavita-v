"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
# from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn.tree import DecisionTreeClassifier
from linearRegression.linearRegression import LinearRegression


########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'entropy'
tree = DecisionTreeClassifier(criterion=criteria)
# Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
# Classifier_B.fit(X, y)
# y_hat = Classifier_B.predict(X)
# [fig1, fig2] = Classifier_B.plot()
# print('Criteria :', criteria)
# print('Accuracy: ', accuracy(y_hat, y))
# for cls in y.unique():
#     print('For class '+str(cls))
#     print('Precision: ', precision(y_hat, y, cls))
#     print('Recall: ', recall(y_hat, y, cls))


X1 = pd.DataFrame([[1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8],
                  [2,1], [2,2], [2,3], [2,4], [2,5], [2,6], [2,7], [2,8],
                  [3,1], [3,2], [3,3], [3,4], [3,5], [3,6], [3,7], [3,8],
                  [4,1], [4,2], [4,3], [4,4], [4,5], [4,6], [4,7], [4,8],
                  [5,1], [5,2], [5,3], [5,4], [5,5], [5,6], [5,7], [5,8],
                  [6,1], [6,2], [6,3], [6,4], [6,5], [6,6], [6,7], [6,8],
                  [7,1], [7,2], [7,3], [7,4], [7,5], [7,6], [7,7], [7,8],
                  [8,1], [8,2], [8,3], [8,4], [8,5], [8,6], [8,7], [8,8]])
y1 = pd.Series([1,1,1,1,1,0,0,0,
               1,1,1,1,1,0,0,0,
               1,1,0,1,1,0,0,0,
               1,1,1,1,1,0,0,0,
               1,1,1,1,1,0,0,1,
               0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0,
               0,0,0,0,0,0,0,0,])

Bagging = BaggingClassifier(base_estimator=tree, n_estimators=5)
Bagging.fit(X1,y1)
[fig3, fig4] = Bagging.plot()
