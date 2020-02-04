import numpy as np
import pandas as pd


def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    total = len(y)
    accurate = 0.0
    for i in range(total):
        if y_hat[i] == y[i]:
            accurate += 1
    accuracy = accurate/total

    return accuracy


def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    total = 0.0
    precise = 0.0
    for i in range(len(y)):
        if y_hat[i] == cls:
            if y[i] == cls:
                precise += 1
            total += 1
    if total==0:
        return 0.0
    precision = precise/total
    
    return precision


def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    total = 0.0
    marked = 0.0
    for i in range(len(y)):
        if y[i] == cls:
            if y_hat[i] == cls:
                marked += 1
            total += 1
    recall = marked/total

    return recall

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    sum = 0.0
    for i in range(len(y)):
        sum += (y_hat[i] - y[i]) ** 2
    rmse = (sum/len(y))**(1/2)

    return rmse

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    abs_sum = 0.0
    for i in range(len(y)):
        abs_sum += abs(y_hat[i] - y[i])
    mae = abs_sum / len(y)

    return mae
