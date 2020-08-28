import matplotlib.pyplot as plt
import pandas
import numpy as np
import pywt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import sklearn.preprocessing
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from fastdtw import fastdtw
import math
from datetime import datetime

''' plot all of the data passed individualy and against each other '''
def plot_all_data(data_names, data_values, time):
    n = len(data_values)
    plt.figure()
    plt.suptitle("Plot of all data")
   
    for i in range(0, n):
        plt.subplot(n, 1, i+1)
        plt.plot_date(time, data_values[i], markersize=1)
        plt.title(data_names[i])

    plt.figure()
    plt.suptitle("Comparison for all data")
    for i in range(0, n):
        for j in range(0, n):
            plt.subplot(n, n, i*4+j+1)
            plt.scatter(data_values[i], data_values[j], s=10)
            plt.xlabel(data_names[i])
            plt.ylabel(data_names[j])


''' replace all nan values in x with replace '''
def correct_missing_values(x, replace):
    for i in range(0, x.size):
        if(math.isnan(x[i])):
            x[i] = replace
    return x

''' remove all nan values in y and corrosponding x '''
def remove_missing_values(x, y):
    retX = []
    retY = []
    i = 0
    while(i < len(y)):
        if(math.isnan(y[i]) == False):
            retX.append(x[i])
            retY.append(y[i])
        i+=1
    return retX, np.array(retY)

''' discrete wavelet analysis to level n on y
if plot then a plot is generated
if rand then the noise component is returned: x, y
otherwise returns: xLowFreq, yLowFreq, xHighFreq, yHighFreq(1), allCoefficints
'''
def dwa(x, y, n, plot=False, rand=False):
    coef = pywt.wavedec(y, 'db1', level=n)
    if (math.isnan(coef[0][0]) or coef[0][0]==0):
        cA = coef[0]
    else:
        cA = coef[0]*((y[0]/coef[0][0]))
        print("IT happened")
        
    filteredX = np.linspace(np.min(x), np.max(x), cA.size)

    if(plot):
        fig = plt.figure()   
        fig.suptitle('n level dwt, y')

        plt.subplot(n+2,1,1)
        plt.title("y")
        plt.plot(x, y)

        plt.subplot(n+2,1,2)
        plt.title("cA")
        plt.plot(filteredX, cA)

        for i in range (1, n+1):
            plt.subplot(n+2, 1, i+2)
            title = 'cD' + str(i)
            relX = np.linspace(np.min(x), np.max(x), coef[i].size)
            plt.plot(relX, coef[i])
            plt.title(title)

        plt.tight_layout()
        

    if rand:
        relX = np.linspace(np.min(x), np.max(x), coef[n].size)
        return relX, coef[n] 
    else:
        # print(coef)
        relX = np.linspace(np.min(x), np.max(x), coef[1].size)
        return filteredX, cA, relX, coef[1], coef

''' create a matrix of x with lag number of previous values '''
def apply_lag(x, lag):
    ret = x
    for i in range(1, lag):
        x = np.c_[x, ret.shift(i)]
    return x

''' mask False if not inlier (y - pred > threshold), true otherwise '''
def get_mask_from_threshold(y, pred, threshold):
    y = np.array(y).ravel()
    count = 0
    mask = np.zeros(len(y))
    for i in range(0, len(y)):
        if abs(y[i] - pred[i]) > threshold:
            mask[i] = False
            count+=1
        else:
            mask[i] = True
    return mask, count

''' apply mask to data so that outlier values (mask False) are nan '''
def apply_mask(data, mask):
    # must be a buit in 
    maskedY = np.zeros(len(data))
    for i in range(0, len(data)):
        if mask[i]:
            maskedY[i] = data[i]
        else:
            maskedY[i] = float('nan')
    return maskedY


def apply_mask_outlier(data, mask):
    # must be a buit in 
    maskedY = np.zeros(len(data))
    outliersY = np.zeros(len(data))
    for i in range(0, len(data)):
        if mask[i]:
            maskedY[i] = data[i]
            outliersY[i] = float('nan')
        else:
            maskedY[i] = float('nan')
            outliersY[i] = data[i]
    return maskedY, outliersY


