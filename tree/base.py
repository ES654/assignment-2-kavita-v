"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth=1):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        pass

    def fit(self, X, y, w):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        w: sample weights
        """
        # works for decision stump
        igs=[]
        points=[]
        split_inds=[]
        features = X.columns.values.tolist()
        for feature in features:
            attr = X[feature]
            ig, point, split_ind = information_gain(y, w, attr)
            igs.append(ig)
            points.append(point)
            split_inds.append(split_ind)
        ind = np.argmax(igs)
        split_attr = features[ind]
        split_ind = split_inds[ind]
        X1 = X.copy()
        X1['y'] = y
        X1.sort_values(by=split_attr, inplace=True)
        y_split = X1['y']
        lt_val = y_split[:split_ind].value_counts().idxmax()
        rt_val = y_split[split_ind:].value_counts().idxmax()

        self.split = (split_attr, point, lt_val, rt_val)
        # print(split_attr, point, lt_val, rt_val)
        return self


        # pass

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        attr = self.split[0]
        val = self.split[1]
        lt_val = self.split[2]
        rt_val = self.split[3]

        attr_col = X[attr]
        y=[]
        for i in range(len(attr_col)):
            if attr_col.values[i] <= val:
                y.append(lt_val)
            else:
                y.append(rt_val)

        return pd.Series(y)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
