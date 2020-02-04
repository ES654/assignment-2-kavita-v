
import numpy as np 
import pandas as pd
from random import randrange
from statistics import mode
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree
import copy

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        models = []
        bags_X = []
        bags_y = []
        for i in range(self.n_estimators):
            bag_X = []
            bag_y = []
            X_copy = X.values
            y_copy = y.values
            while len(bag_X) < len(X):
                index = randrange(len(X_copy))
                bag_X.append(X_copy[index])
                bag_y.append(y_copy[index])
            model = self.base_estimator
            model.fit(bag_X,bag_y)
            # print(tree.export_graphviz(model))
            models.append(copy.deepcopy(model))
            bags_X.append(bag_X)
            bags_y.append(bag_y)

        self.models = models
        self.X = X
        self.y = y
        self.bags_X = bags_X
        self.bags_y = bags_y
        return self

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred = []
        for i in range(len(X)):
            preds=[]
            X_test = X[X.index==i]
            for model in self.models:
                pred = model.predict(X_test)
                preds.append(pred[0])
            y_pred.append(mode(preds))
        return pd.Series(y_pred)

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        h = .02
        i=1
        bags_X = self.bags_X
        bags_y = self.bags_y
        fig1 = plt.figure(figsize=(45, 9))

    
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
       
        for model in self.models:
            ax = plt.subplot(1, len(self.models) , i)
            X = pd.DataFrame(bags_X[i-1])
            y = pd.Series(bags_y[i-1])
            x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
            y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = np.array(model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
            # print(Z[12])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            ax.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap=cm_bright, edgecolors='k')
            # size=[1000*w for w in self.weights[i-1]]
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(str(X.columns[0]))
            ax.set_ylabel(str(X.columns[1]))
            plt.title("Estimator "+str(i))
            i+=1
            
        fig2 = plt.figure(figsize=(9,9))
        X = self.X
        y = self.y
        ax2 = plt.subplot(1,1,1)
        x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
        y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # size=[1000*w for w in self.weights[i-2]]
        ax2.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap=cm_bright, edgecolors='k')
        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())
        plt.title("Combined Decision Surface")
        
        plt.tight_layout()
        plt.show()

        return [fig1,fig2]
