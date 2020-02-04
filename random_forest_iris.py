import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn import datasets
from sklearn.utils import shuffle

np.random.seed(42)

###Write code here
################ Iris Dataset ##########

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=[0,1,2,3])
X = iris_df
y = iris.target
for i in range(len(y)):
    if y[i]==2: #virginica
        y[i]=1
    else:       #not-virginica
        y[i]=0
# X = X[['sepal width (cm)', 'petal width (cm)']]
X, y = shuffle(X, y, random_state=42)
spl = int(len(X)*0.6)
X_train = X[:spl]
X_test = X[spl:]
y_train = y[:spl]
y_test = pd.Series(y[spl:])

randomforest = RandomForestClassifier(3, criterion = 'entropy', max_depth=5)
randomforest.fit(X_train,y_train)
y_hat = randomforest.predict(X_test)
randomforest.plot()
