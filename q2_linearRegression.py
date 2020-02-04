import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    print("fit_intercept="+str(fit_intercept))
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = LR.predict(X)
    LR.plot_residuals()

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))





X3 = pd.DataFrame(np.random.randn(50, 4))
y3 = pd.Series(np.random.randn(50)) 
LR3 = LinearRegression(fit_intercept=False)
start3=time.time()
LR3.fit(X3,y3)
end3=time.time()
print("N=50, P=4, time="+str(end3-start3))

X2 = pd.DataFrame(np.random.randn(100, 2))
y2 = pd.Series(np.random.randn(100)) 
LR2 = LinearRegression(fit_intercept=False)
start2=time.time()
LR2.fit(X2,y2)
end2=time.time()
print("N=100, P=2, time="+str(end2-start2))

X1 = pd.DataFrame(np.random.randn(50, 2))
y1 = pd.Series(np.random.randn(50)) 
LR1 = LinearRegression(fit_intercept=False)
start1=time.time()
LR1.fit(X1,y1)
end1=time.time()
print("N=50, P=2, time="+str(end1-start1))
