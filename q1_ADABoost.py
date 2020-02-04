"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression
from sklearn.utils import shuffle
np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 40
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))

y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria, max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('For class '+str(cls))
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
X = iris_df
y = iris.target
for i in range(len(y)):
    if y[i]==2: #virginica
        y[i]=1
    else:       #not-virginica
        y[i]=0
X = X[['sepal width (cm)', 'petal width (cm)']]
X, y = shuffle(X, y, random_state=42)
spl = int(len(X)*0.6)
X_train = X[:spl]
X_test = X[spl:]
y_train = y[:spl]
y_test = pd.Series(y[spl:])


adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
stump = DecisionTree(criterion=criteria, max_depth=1)

adaboost.fit(X_train,y_train)
stump.fit(X_train,y_train,[(1/len(y_train)) for i in y_train])

y_hat_adaboost = adaboost.predict(X_test)
y_hat_stump = stump.predict(X_test)

[fig3, fig4] = adaboost.plot()



print('Classifier: Adaboost')
print('Accuracy: ', accuracy(y_hat_adaboost, y_test))
for cls in y_test.unique():
    print('For class '+str(cls))
    print('Precision: ', precision(y_hat_adaboost, y_test, cls))
    print('Recall: ', recall(y_hat_adaboost, y_test, cls))

print('Classifier: Decision Stump')
print('Accuracy: ', accuracy(y_hat_stump, y_test))
for cls in y_test.unique():
    print('For class '+str(cls))
    print('Precision: ', precision(y_hat_stump, y_test, cls))
    print('Recall: ', recall(y_hat_stump, y_test, cls))
