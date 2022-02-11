from typing import List
from attr import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import root
from torch import less
import sklearn
from .utils import entropy, information_gain, gini_index



class DecisionTree():
    def __init__(self, criterion = 'information_gain', max_depth=np.inf):
        """
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """

        # gini score ~~ info gain except that entropy replaced with gini index
        if criterion == "information_gain":
            self.criterion = entropy
        else:
            self.criterion = gini_index

        self.maxDepth = max_depth
        self.tree: dict = None

    def DIDO(self, examples: pd.DataFrame, targets: pd.Series, validNextItAttr, depth=0, rootAttr=None):

        if len(targets.drop_duplicates()) == 1 or len(examples.drop_duplicates()) == 1 or depth >= self.maxDepth:
            return targets.value_counts().idxmax()
        elif len(validNextItAttr) == 0  or len(targets) == 0:  # incase of no more attributes to split on
            return rootAttr

        rootAttr = self.getRoot(examples, targets, validNextItAttr)
        node = {rootAttr: {}}
        splitExamples = examples[rootAttr]
        allAttributes = splitExamples.unique()
        validNextItAttr = [attr for attr in allAttributes if attr != rootAttr]
        for valueOfAttribute in allAttributes:
            subsetIndices = (splitExamples == valueOfAttribute)
            subExamples = examples.iloc[np.where(subsetIndices)]
            subTargets = targets.iloc[np.where(subsetIndices)]
            if len(subExamples) == 0:
                node[rootAttr][valueOfAttribute] = targets.value_counts().idxmax()
            else:
                node[rootAttr][valueOfAttribute] = self.DIDO(subExamples, subTargets, validNextItAttr, depth + 1, rootAttr)

        return node

    def DIRO(self, examples: pd.DataFrame, targets: pd.Series, validNextItAttr, depth=0, rootAttr=None):

        if len(examples.drop_duplicates()) == 1 or depth >= self.maxDepth:
            return targets.mean()
        elif len(validNextItAttr) == 0 or len(targets) == 0:  # incase of no more attributes to split on
            return rootAttr

        rootAttr = self.getRoot(examples, targets, validNextItAttr)
        node = {rootAttr: {}}
        splitExamples = examples[rootAttr]
        allAttributes = splitExamples.unique()
        validNextItAttr = [attr for attr in allAttributes if attr != rootAttr]

        for valueOfAttribute in allAttributes:
            subsetIndices = (splitExamples == valueOfAttribute)
            subExamples = examples.iloc[np.where(subsetIndices)]
            subTargets = targets.iloc[np.where(subsetIndices)]
            if len(subExamples) == 0:
                node[rootAttr] = targets.mean()
            else:
                node[rootAttr][valueOfAttribute] = self.DIRO(subExamples, subTargets, validNextItAttr, depth + 1, rootAttr)

        return node


    def getRootRI(self, examples : pd.DataFrame, targets: pd.Series):
        # TODO
        ent = self.criterion(targets)
        gains = []
        for col in examples.columns:
            sortedColumn = examples[col].copy().sort_values(ascending= True)
            indices = sortedColumn.index
            newTargets = targets.loc[indices]
            newTargets.reset_index(drop=True, inplace=True)
            
            for split in range(0, len(examples)):
                leftRange = newTargets.iloc[:split+1]
                rightRange = newTargets.iloc[split+1:]

                entLeft = self.criterion(leftRange) * (len(leftRange) / len(targets))
                entRight = self.criterion(rightRange) * (len(rightRange) / len(targets))

                gain = ent - (entLeft + entRight)

                gains.append([gain, col, split])

        highestGain, bestCol, splitInd = sorted(gains, key=lambda x: x[0], reverse=True) [0]
        thresh = (examples[bestCol].iloc[splitInd] + examples[bestCol].iloc[splitInd - 1])/2.

        return bestCol, thresh


    def RIDO(self, examples: pd.DataFrame, targets: pd.Series, depth=0, rootAttr=None):

        if len(examples.drop_duplicates()) == 1 or depth >= self.maxDepth:
            return targets.value_counts().idxmax()

        rootAttr, thresh = self.getRootRI(examples, targets)

        lessIndices = examples[rootAttr] <= thresh
        lessExamples = examples.iloc[np.where(lessIndices)]
        lessTargets = targets.iloc[np.where(lessIndices)]

        moreIndices = examples[rootAttr] > thresh
        moreExamples = examples.iloc[np.where(moreIndices)]
        moreTargets = targets.iloc[np.where(moreIndices)]


        if len(lessTargets) == 0:
            return moreTargets.value_counts().idxmax()
        elif len(moreTargets) == 0:
            return lessTargets.value_counts().idxmax()


        node = {
            f'{rootAttr},{thresh}' :{ #f'{rootAttr} <= {thresh}':
                'Y': self.RIDO(lessExamples, lessTargets, depth + 1, rootAttr), 
                'N': self.RIDO(moreExamples, moreTargets, depth + 1, rootAttr)
            }
        }
        return node

    def RIRO(self, examples: pd.DataFrame, targets: pd.Series, depth=0, rootAttr=None):

        if len(examples.drop_duplicates()) == 1 or depth >= self.maxDepth:
            return targets.mean()

        rootAttr, thresh = self.getRootRI(examples, targets)

        lessIndices = examples[rootAttr] <= thresh
        lessExamples = examples.iloc[np.where(lessIndices)]
        lessTargets = targets.iloc[np.where(lessIndices)]

        moreIndices = examples[rootAttr] > thresh
        moreExamples = examples.iloc[np.where(moreIndices)]
        moreTargets = targets.iloc[np.where(moreIndices)]


        if len(lessTargets) == 0:
            return moreTargets.mean()
        elif len(moreTargets) == 0:
            return lessTargets.mean()


        node = {
            f'{rootAttr},{thresh}' :{ #f'{rootAttr} <= {thresh}':
                'Y': self.RIRO(lessExamples, lessTargets, depth + 1, rootAttr), 
                'N': self.RIRO(moreExamples, moreTargets, depth + 1, rootAttr)
            }
        }
        return node


    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if X.dtypes[0].name != 'category':
            if y.dtype.name != 'category':
                self.criterion = np.var
                self.tree = self.RIRO(X, y)
                self.type = 'RIRO'
            else:
                self.tree = self.RIDO(X, y)
                self.type = 'RIDO'

        elif X.dtypes[0].name == 'category':
            if y.dtype.name != 'category':
                self.criterion = np.var
                self.tree = self.DIRO(X, y, X.columns)
                self.type = 'DIRO'
            else:
                self.tree = self.DIDO(X, y, X.columns)
                self.type = 'DIDO'

    def getRoot(self, X, y, validNextAttr):
        gains = []
        for attr in validNextAttr:
            gain = information_gain(y, X[attr], self.criterion)
            gains.append(gain)

        bestAttr = validNextAttr[np.argmax(gains)]

        return bestAttr

    def predictDiscrete(self, X):
        pred = []
        for i in range(len(X)):
            row = X.iloc[i]

            tree = self.tree.copy()
            while type(tree) == dict:
                rootAttr = list(tree.keys())[0]
                rootAttrValue = row[rootAttr]
                tree = tree[rootAttr][rootAttrValue]

            pred.append(tree)

        return pd.Series(pred)


    def predictReal(self, X):
        pred = []
        for i in range(len(X)):
            row = X.iloc[i]

            tree = self.tree.copy()
            while type(tree) == dict:
                
                key = list(tree.keys())[0]
                rootAttr, thresh = key.split(',')
                rootAttr, thresh = int(rootAttr), float(thresh)
                rootAttrValue = row.loc[rootAttr]
                tree = tree[key]
                if rootAttrValue <= thresh:
                    tree = tree['Y']
                else:
                    tree = tree['N']

            pred.append(tree)
        ret = pd.Series(pred)
        return ret


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        if self.type in ['RIDO', 'RIRO']:
            return self.predictReal(X)
        else:
            return self.predictDiscrete(X)

        
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

       

        if self.type == 'RIDO':
            def plotTree(tree, depth=0):
                if type(tree) == dict:
                    for rootAttr in list(tree.keys()):
                        condn = rootAttr.replace(',', '<=')
                        stringval = f'?(X{condn})' + '\n' + '\t' * (depth + 1) + 'Y: ' + \
                            plotTree(tree[rootAttr]['Y'], depth + 1) + \
                            '\n' + '\t' * (depth + 1) + 'N: ' + \
                            plotTree(tree[rootAttr]['N'], depth + 1)
                else:
                    return f'Class {tree}'
                return stringval
            print(plotTree(self.tree))
        elif self.type == 'RIRO':
            def plotTree(tree, depth=0):
                if type(tree) == dict:
                    for rootAttr in list(tree.keys()):
                        condn = rootAttr.replace(',', '<=')
                        stringval = f'?(X{condn})' + '\n' + '\t' * (depth + 1) + 'Y: ' + \
                            plotTree(tree[rootAttr]['Y'], depth + 1) + \
                            '\n' + '\t' * (depth + 1) + 'N: ' + \
                            plotTree(tree[rootAttr]['N'], depth + 1)
                else:
                    return f'{tree}'
                return stringval
            print(plotTree(self.tree))

        elif self.type in ['DIRO', 'DIDO']:
            def plotTree(tree, depth=0):
                if type(tree) == dict:
                    stringval = ''
                    for rootAttr in list(tree.keys()):

                        subtree = tree[rootAttr]

                        for possibleVal in list(subtree.keys()):
                            substringval = f'?(X{rootAttr} == {possibleVal})' + '\n' + '\t' * (depth + 1)
                            substringval += plotTree(subtree[possibleVal], depth + 1) #+ \
                            stringval += '\n' + '\t' * (depth) + substringval
                else:
                    return f'{tree}'
                return stringval
            print(plotTree(self.tree))
