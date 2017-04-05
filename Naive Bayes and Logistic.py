import numpy as np
from numpy import genfromtxt
import math
import random
from collections import defaultdict

'''This function caluclates sigmoid of theta multiplied by X'''
def logistic_func(theta, x):
    """This function calculates sigmoid.

    Args:
        theta: Value of theta (W)
        X: Input array
    Returns:
        numpy float: sigmoid value

    """
    return float(1) / (1 + math.e**(-x.dot(theta.T)))

def process_data(data,size):
    """This function processes the data. Removes missing data and splits the data into training and testing

       Args:
           Unprocessed data
       Returns:
           numpy array: training and testing data set

    """
    data = np.delete(data, 0, 1)
    data = data[~np.isnan(data).any(axis=1)]
    X = data[:, 0:data.shape[1] - 1]
    Y = data[:, data.shape[1] - 1: data.shape[1]]
    train_size = int(X.shape[0] * 2 / 3)
    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_test = X[train_size:X.shape[0] + 1]
    Y_test = Y[train_size:X.shape[0] + 1]
    randomData = np.append(X_train, Y_train, axis=1)
    np.random.shuffle(randomData)
    fractionSize = int(randomData.shape[0] * size)
    X_train = randomData[0:fractionSize, 0:9]
    Y_train = np.array(randomData[0:fractionSize, 9])
    return X_train, X_test, Y_train, Y_test

def logistic_train(X_train, Y_train):
    """This function trains the logistic regression on  training data and returns W

        Args:
             training data and labels.

        Returns:  W

    """
    X_train = np.insert(X_train, 0, 1, axis=1)
    Y_train = np.where(Y_train == 2, 1, 0)
    feature_size = X_train.shape[1]
    W = np.random.rand(feature_size)
    learnin_rate = 0.0001
    iter_count = 0
    while True:
        probY = logistic_func(W, X_train)
        cost = np.dot(np.subtract(Y_train.T, probY.T), X_train)
        prev_W = W
        W = W + learnin_rate * cost
        dif = np.subtract(prev_W, W)
        if (dif < 0.0001).all() or iter_count >= 10000:
            break
        iter_count += 1
    return W

def logistic_test(X_test, Y_test, W):
    """This function tests the trained  logistic regression on  test data and returns accurcay

        Args:
            testing data, labels and W.

        Returns:  Accuracy

    """
    X_test = np.insert(X_test, 0, 1, axis=1)
    Y_test = np.where(Y_test == 2, 1, 0)
    probTest = logistic_func(W, X_test)
    predTest = np.where(probTest >= .5, 1, 0)
    correct = 0
    for i in range(0, len(Y_test)):
        if predTest[i] == Y_test[i]:
            correct += 1
    test_acc = 100 * correct / (1.0 * len(Y_test))
    return test_acc

def bayes_train(X_train,Y_train):
    """This function trains the Naive Bayes  on  training data and returns probabilities

           Args:
                training data and labels.

           Returns:  Probabilities

    """
    Y_train = np.where(Y_train== 2, 1, -1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    XconcatY = np.append(X_train, Y_train, axis=1)
    filterPos = np.array(XconcatY[XconcatY[:, XconcatY.shape[1] - 1] == 1])
    filterNeg = np.array(XconcatY[XconcatY[:, XconcatY.shape[1] - 1] == -1])
    probPos = filterPos.shape[0]
    probNeg = filterNeg.shape[0]
    dictPosProb = {}
    dictNegProb = {}
    for i in xrange(0,9):
        unique, counts = np.unique(filterPos[:,i], return_counts=True)
        tempPos=defaultdict(int, dict(zip(unique, counts)))
        for j in range(1,10):
            if tempPos[j]==0:
                tempPos[j]=1
            else:
                tempPos[j]=tempPos[j]+1
        tempPos.update((k, 1.0 * v /(probPos+1)) for k, v in tempPos.items())
        dictPosProb[i]=tempPos
        unique, counts = np.unique(filterNeg[:, i], return_counts=True)
        tempNeg = defaultdict(int, dict(zip(unique, counts)))
        for j in range(1,10):
            if tempNeg[j]==0:
                tempNeg[j]=1
            else:
                tempNeg[j]=tempNeg[j]+1
        tempNeg.update((k, 1.0 * v / (probNeg+1)) for k, v in tempNeg.items())
        dictNegProb[i] = tempNeg
    return probPos,probNeg,dictPosProb ,dictNegProb

def bayes_test(probPos, probNeg, dictPosProb, dictNegProb,X_test,Y_test):
    """This function tests the Naive Bayes  on  testing data and returns accuracy

               Args:
                    testing data and probabilities.

               Returns:  Accuracy

    """
    Y_test = np.where(Y_test == 2, 1, -1)
    predY = np.zeros(X_test.shape[0])
    list = [1, -1]
    for i in xrange(X_test.shape[0]):
        posC = probPos
        negC = probNeg
        for j in xrange(X_test.shape[1]):
            posC *= dictPosProb[j][X_test[i, j]]
            negC *= dictNegProb[j][X_test[i, j]]
            if posC > negC:
                predY[i] = 1
            elif posC < negC:
                predY[i] = -1
            else:
                predY[i]=random.choice(list)
    correct = 0
    for i in range(0, len(Y_test)):
        if predY[i] == Y_test[i]:
            correct += 1
    test_acc = 100 * correct / (1.0 * len(Y_test))
    return test_acc

if __name__=="__main__":
    data = genfromtxt('breast-cancer-wisconsin.data.txt', delimiter=',')
    datasize=[0.01,0.02,0.03,0.125,0.625,1]
    logist_accuracy = {}
    bayes_accuracy={}
    for size in datasize:
        X_train, X_test, Y_train, Y_test=process_data(data,size)
        W = logistic_train(X_train,Y_train)
        logist_accuracy[size]=logistic_test(X_test, Y_test, W)
        probPos, probNeg, dictPosProb, dictNegProb=bayes_train(X_train,Y_train)
        bayes_accuracy[size]=bayes_test(probPos, probNeg, dictPosProb, dictNegProb,X_test,Y_test)

    print "Accuracy with logistic regression:"
    for key in sorted(logist_accuracy.keys()):
        print "Accuracy of testing data using logistic regresion with "+repr(key)+" fraction of training data is "+repr(logist_accuracy[key])
    print ""
    print "Accuracy with Naive Bayes:"
    for key in sorted(bayes_accuracy.keys()):
        print "Accuracy of testing data using naive bayes  with " + repr(key) + " fraction of training data is " + repr(bayes_accuracy[key])
