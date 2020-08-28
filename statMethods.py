import matplotlib.pyplot as plt
import numpy as np
import pandas
from tools import * 
from scipy import stats
from datetime import datetime


def iqr_removal(x, y, threshold, plot=False):
    y = correct_missing_values(y, np.nanmean(y))
    smallQuantile = np.percentile(y, 25)
    largeQuantile = np.percentile(y, 75)
    iqr = largeQuantile - smallQuantile
    threshole = iqr*threshold
    smallRange = smallQuantile - threshold
    largeRange = largeQuantile + threshold
    print(smallRange, largeRange)
    count = 0
    outliers = np.zeros(len(x))
    outliers = outliers + (-2)
    
    for i in range(0, len(y)):
        if ((y[i] > largeRange) or (y[i] < smallRange)):
            outliers[i] = y[i]
            y[i] = float('nan')
            count += 1

    print("iqr removal found ", count, " outliers",)

    if (plot) :
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, outliers, '.r')
        plt.title("iqr")
        # plt.show()

    return y

def z_score_removal(x, y, threshold, plot=False):
    # must be gaussian
    count = 0
    print(np.nanmean(y))
    y = correct_missing_values(y, np.nanmean(y))
    scores = stats.zscore(y)
    outliers = np.zeros(len(x))
    outliers = outliers + (-1000)
    for i in range(0, len(y)):
        if (scores[i] > threshold):
            outliers[i] = y[i]
            y[i] = float('nan')
            count += 1

    print("z score removal found ", count, " outliers")
    
    if (plot) :
        plt.plot(x, y, '.')
        plt.plot(x, outliers, '.r')
       
    return y


def rolling_mean_removal(x, y, window, threshold, plot=False):
    data = pandas.DataFrame(y)
    rolled = data.rolling(window).mean()

    
    count = 0
    outliers = np.zeros(len(x))
    outliers = outliers + (-10)
    
    difference = data.values - rolled.values
    for i in range(0, len(difference)):
        if (np.isnan(difference[i])):
            difference[i] = 0
        elif (difference[i] > threshold):
            outliers[i] = y[i]
            y[i] = float('nan')
            count += 1
            # print("Data point ", i, " removed")

    print("rolling mean removal found ", count, " outliers")

    if (plot) :
        plt.figure()
        plt.plot(x, y, '.')
        plt.plot(x, outliers, '.r')
        plt.plot(x, rolled, alpha=0.5)
        plt.title("Rolling mean")

    return y



