from numpy.linalg import pinv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept = fit_intercept
        self.method=method

        pass

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        self.X=X
        self.y=y
        if self.fit_intercept==True:
            X = np.hstack((np.ones((len(X),1)),X))
        thetha = np.dot(pinv(np.dot(X.T,X)),np.dot(X.T, y))
        self.thetha = thetha
        return self

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        y_pred=[]
        if self.fit_intercept==True:
            X = np.hstack((np.ones((len(X),1)),X))
        else:
            X = X.values
        for i in range(len(X)):
            data = X[i]
            pred = np.dot(self.thetha,data.T)
            y_pred.append(pred)
        return pd.Series(y_pred)

    def plot_residuals(self):
        '''
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)

        '''
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
        y = self.y
        y_hat = self.predict(self.X)
        ax1.scatter(y,y_hat)
        ax1.set_xlabel("Ground truth")
        ax1.set_ylabel("Estimate")
        ax2.hist(y-y_hat)
        ax2.set_xlabel("Residuals")
        var = round(np.var(y-y_hat),3)
        mean = round(np.mean(y-y_hat),3)
        ax2.set_title("Variance:"+str(var)+"     Mean:"+str(mean))
        ax3.bar(["thetha_"+str(i) for i in range(len(self.thetha))],self.thetha, log=True)
        plt.show()
        pass
