
import numpy as np
import pandas as pd
import numpy as np


def entropy(Y : pd.Series):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    tol = 1e-100
    pr = (Y.value_counts() + tol)/(len(Y) + tol) #adding small value to prevent divide by zero errors
    entropy = -np.dot(pr, np.log2(pr))
    return entropy

        
def gini_index(Y: pd.Series):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    tol = 1e-100
    pr = (Y.value_counts() + tol)/(len(Y) + tol)
    index = 1 - np.dot(pr, pr)
    return index
    

def information_gain(Y, attr, eval = entropy):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """

    IG = eval(Y)
    data = pd.DataFrame({'attr': attr, 'Y': Y})
    for v in set(attr):
        subset = data[data['attr']==v]['Y']
        IG -= (len(subset)/len(Y))*eval(subset)  

    return IG
