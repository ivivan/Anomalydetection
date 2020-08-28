## optamising and visualising local outlier factor method

import matplotlib.pyplot as plt
import numpy as np
import pandas
from loadData import *
from tools import *
from sklearn.neighbors import LocalOutlierFactor 

index, time, conductivity, turbidity, nitrateMg, nitrateUm = load_c_data()

''' local outlier factor '''
labels = []
removalPairs = [] # [inliers, outliers]
cont = ['auto', 'auto', 'auto'] # contamination for each k in order
outlierCount = []
ks = [28, 56, 112]

# do detection for each k, later plotted
for j in range(0, len(ks)):
    inliers = []
    k = ks[j]
    data = nitrateMg.copy()
    time, inliers = remove_missing_values(time, data) 
    localFactorDetection = LocalOutlierFactor(n_neighbors=k, contamination=cont[j])
    pred = localFactorDetection.fit_predict(inliers.reshape(-1,1)) 
    #1 is an inlier, -1 is an outlier

    count = 0
    outliers = np.zeros(len(inliers)) + float('nan') 
    # using pred as mask, seperate outliers and inliers
    for i in range(0, len(pred)):
        if (pred[i] == -1):
            outliers[i] = inliers[i]
            inliers[i] = float('nan')
            count += 1

    outlierCount.append(count)
    removalPairs.append([inliers, outliers, time])
    labels.append('%d neighbours. Outliers: %d (%0.1f%%)' % 
            (k, count, 100*float(count)/len(data)))

# plot comparison of different number of outliers   
plt.figure()
for i in range(0, len(labels)):
    plots = len(labels)
    plt.subplot(plots, 1, i+1)
    plt.scatter(removalPairs[i][2], removalPairs[i][0], s=1, c='g')
    plt.scatter(removalPairs[i][2], removalPairs[i][1], s=1.2, c='r')
    plt.title(labels[i])
    plt.ylabel('nitrate levels (mg/L)')
plt.subplots_adjust(hspace=0.35)

# plot only the results of 110 neighbour run, which was determined to be best
plt.figure()
plt.scatter(removalPairs[2][2], removalPairs[2][1], s=0.2, c='r')
plt.plot_date(removalPairs[2][2], removalPairs[2][0], '-', color='k')
plt.ylabel("nitrate levels (mg/L)")
plt.title("Outlier detection with %d neightbors, found %d outliers" % 
                (ks[2], outlierCount[2]))

plt.show()