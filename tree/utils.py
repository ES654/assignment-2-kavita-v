import pandas as pd
import numpy as np


def entropy(Y,w):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    # counts = Y.value_counts(normalize=True)
    # prob_list = list(counts)
    #working for binary
    clsA = Y[0]
    # for i in range(len(Y)):
    #     if Y[i]!=clsA:
    #         clsB=Y[i]
    #         break
    if len(np.unique(Y))==1:
        return 0
    p_A = 0
    p_B = 0
    for i in range(len(Y)):
        if Y[i]==clsA:
            p_A += w[i]
        else:
            p_B += w[i]
    entropy = -(p_A*(np.log2(p_A)))-(p_B*(np.log2(p_B)))
   

    return entropy

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    pass

def information_gain(Y, w, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    df = {'attr':attr, 'Y':Y, 'w':w}
    df = pd.DataFrame(df)
    df = df.sort_values(by="attr")
    Y = df['Y'].values
    w = df['w'].values
    attr = df['attr'].values 
    candidates = []
    igs = []
    inds = []
    for i in range(1,len(Y)):
        if Y[i] != Y[i-1]:
            candidate = float((attr[i]+attr[i-1])/2)
            e1 = entropy(Y[:i], w[:i])
            e2 = entropy(Y[i:], w[i:])
            ig = entropy(Y,w) - (i/len(Y))*e1 - (1 - (i/len(Y)))*e2
            igs.append(ig)
            candidates.append(candidate)
            inds.append(i)
    
    ind = np.argmax(igs)  

    return (igs[ind], candidates[ind], inds[ind])
