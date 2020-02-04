import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from random import randrange

df = pd.read_excel (r'/home/kavita-v/Desktop/assignment-2-kavita-v-master/Real estate valuation data set.xlsx')
y = df['Y house price of unit area']
X = df.drop(columns={'Y house price of unit area','No'}).values


def cross_validation_split(X,y,no_of_folds):
    X_split = []
    y_split = []
    X_copy = list(X)
    y_copy = list(y)
    fold_size = int(len(X)/no_of_folds)
    for i in range(no_of_folds):
        fold_X = []
        fold_y = []
        while len(fold_X) < fold_size:
            index = randrange(len(X_copy))
            fold_X.append(X_copy.pop(index))
            fold_y.append(y_copy.pop(index))
        X_split.append(fold_X)
        y_split.append(fold_y)
    return X_split, y_split



def cross_validate(X, y, no_of_folds):
    X_split, y_split = cross_validation_split(X,y,no_of_folds)
    for i in range(no_of_folds):
        X_test = X_split[i]
        y_test = pd.Series(y_split[i])
        X_train = []
        y_train = []
        for l in X_split[:i]+ X_split[i+1:]:
            X_train = X_train + l
        for l in y_split[:i]+ y_split[i+1:]:
            y_train = y_train + l
        y_train = pd.Series(y_train)
        LR = LinearRegression(fit_intercept=True)
        LR.fit(X_train, y_train)
        y_hat_test = pd.Series(LR.predict(X_test))
        y_hat_train = pd.Series(LR.predict(X_train))
        mae_test = mae(y_hat_test,y_test)
        mae_train = mae(y_hat_train, y_train)
        LR.plot_residuals()
        print("Fold "+str(i)+":\tTrain MAE="+str(mae_train)+"\tTest MAE="+str(mae_test))

    
cross_validate(X,y,5)
