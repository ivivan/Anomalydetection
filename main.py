from loadData import *
from statMethods import *
from clusteringMethods import *
from regressionMethods import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats.stats import pearsonr
import statsmodels
import pandas
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
index, time, conductivity, turbidity, nitrateMg, nitrateUm = load_c_data()

'''plot all the data '''
# plt.scatter(time, nitrateMg, s=1, c='k')
# plt.ylabel("Nitrate level (mg/L)")
# plt.show()
# data_names = ['conductivity', 'turbidity', 'nitrateMg', 'nitrateUm']
# data_values = [conductivity, turbidity, nitrateMg, nitrateUm]
# plot_all_data(data_names, data_values, time)

''' statistical methods '''
# threshold = np.abs(np.min(turbidity) - np.max(turbidity)) / 5 #5% difference
# threshold = 200

# print(threshold)
# rolling_mean_removal(time, turbidity, 5, threshold, True)

# threshold = 3
# turbidity = z_score_removal(time, turbidity, threshold)

# threshold = 5
# iqr_removal(time, nitrateMg, threshold, True)


# threshold = np.average(nitrateMg) /  (10* np.abs(np.min(nitrateMg) - np.max(nitrateMg)))

''' multidimentional clustering methods '''
# one_class_removal_2d(index, conductivity, nitrateMg, ['conductivity', 'nitrateMG'], True)
# one_class_removal_2d(index, nitrateUm, nitrateMg, ['conductivity', 'nitrateMG'], True)
# one_class_removal_1d(index, conductivity, True)

# local_outlier_removal_2d(index, conductivity, nitrateMg, ['conductivity', 'nitrateMG'], True)
# local_outlier_removal_1d(index, nitrateMg, 'nitrate', True)
# local_outlier_removal_1d(index, conductivity, 'conductivity', True)
# local_outlier_removal_1d(index, turbidity, 'turbidity', True)

''' regression methods '''
# train on months 5-7 and test on 2-5 in 2017
# data = pandas.DataFrame({'time': time, 'nitrateMg' : conductivity, 'nitrateUm' : index})
# data.set_index('time', inplace=True)

# nitrateTrain = data.truncate(before=datetime.strptime("2017-05-01", '%Y-%m-%d'), 
#                             after=datetime.strptime("2017-07-01", '%Y-%m-%d'))
# nitrateTest = data.truncate(before=datetime.strptime("2017-02-01", '%Y-%m-%d'), 
#                             after=datetime.strptime("2017-05-01", '%Y-%m-%d'))
# nitrateTrain = data
# nitrateTest= data

# regress_removal(nitrateTrain['nitrateUm'].values, nitrateTrain['nitrateMg'].values, 
#                 nitrateTest['nitrateUm'].values, nitrateTest['nitrateMg'].values, True)


''' RANSAC '''
# print(index)

# data = pandas.DataFrame({'nitrateMg' : nitrateMg, 'index' : index})
data = pandas.DataFrame({'time': time, 'nitrateMg' : nitrateMg, 'index' : index, 
                        'conductivity': conductivity, 'turbidity' : turbidity, 
                        'nitrateUm': nitrateUm})
data.set_index('time', inplace=True)

# plt.plot(index, np.sin(index/(10)), '-')

nTrain = data.truncate(before=datetime.strptime("2017-7-01", '%Y-%m-%d'), 
                            after=datetime.strptime("2018-1-02", '%Y-%m-%d'))
nTest = data.truncate(before=datetime.strptime("2018-1-2", '%Y-%m-%d'), 
                            after=datetime.strptime("2018-2-01", '%Y-%m-%d'))
# index, nitrateMg  = dwa(nTrain['index'].values, nTrain['nitrateMg'].values, 2, True, True)
# nitrateTrain = pandas.DataFrame({'index': index, 'nitrateMg': nitrateMg})
# index, nitrateMg  = dwa(nTest['index'].values, nTest['nitrateMg'].values, 2, False, True)
# nitrateTest = pandas.DataFrame({'index': index, 'nitrateMg': nitrateMg})


# xTrain, xTest, yTrain, yTest = train_test_split(nTrain['index'].values, nTrain['nitrateMg'].values)
# xTrain, xTest, yTrain, yTest = train_test_split(index, np.sin(index/10))

X = nTrain['index'].values
y = nTrain['nitrateMg'].values

# xTrain, xTest, X1, yTrain, X2, yTest = data_prep(X, y, 3)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest)
# MLP_regression(xTrain, xTest, X1, yTrain, X2, yTest)
# regress_removal(xTrain, yTrain, xTest, yTest, X1, X2, True)

## Comparing different ns
# 30 segments looks better than 3, what would be ideal?

# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, 1)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "One")
# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, 3)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "three")
# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, 50)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "50 without day ")

# print("here")
lag = 50
# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, lag)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "season %s" % lag)

## adding month to all data
# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(index, nitrateMg, lag)
# data['day'] = pandas.DatetimeIndex(data.index).month
# dayData = data['day'].values[0:nitrateMg.size/2-50]
# print(len(dayData), len(xTrain))
# xTrain = np.c_[xTrain, dayData]
# dayTest = data['day'].values[len(nitrateMg)/2-50: len(nitrateMg)-50]
# print(len(dayTest), len(xTest), len(nitrateMg), len(data['day'].values))
# xTest = np.c_[xTest, dayTest]
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "month %s" % lag)

## differencing 
# season =  4*28*12
# diffed = np.diff(nitrateMg, n=season)
# diffed = []
# for i in range(season, len(nitrateMg)):
# 	diffed.append(nitrateMg[i] - nitrateMg[i - season])

# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(index[season:], np.array(diffed), lag)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "differenced month removed %s" % lag)


''' Multi dimentional Regression '''
# y = sklearn.preprocessing.normalize([y], norm='l1').ravel()

# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, lag)
# nTrain['turbidity'] = sklearn.preprocessing.normalize([nTrain['turbidity'].values], norm='l1').ravel()
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "without conductivity")
# condData = nTrain['conductivity'].values[0:y.size/2-50]
# xTrain = np.c_[xTrain, condData]
# condData = nTrain['conductivity'].values[y.size/2-50: y.size-50]
# xTest = np.c_[xTest, condData]

# y1 = sklearn.preprocessing.normalize([nTrain['nitrateMg'].values], norm='l1').ravel()
# y2 = sklearn.preprocessing.normalize([ nTrain['conductivity'].values], norm='l1').ravel()
# plt.plot(nTrain['index'].values, y1, label='nitrate')
# plt.plot(nTrain['index'].values,y2, label='condictivity')
# plt.plot(nTrain['index'].values, y2-y1, label='residuals')
# plt.legend()
# print(y1.ravel())
# print(adfuller(y2-y1))
a, conductivity = remove_missing_values(index, conductivity)
print(conductivity[0:10])
print(adfuller(nitrateMg))
print(kpss(nitrateMg))

# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest, "with conductivity")

# Evaluating whether each method makes sense
# if no relationship, no include
# if high corrolation then include
# if dtw distance is lower than corrolation then include some history
# dwtD = get_dtw_d(nTrain['nitrateMg'], nTrain['conductivity'])
# pcc = pearsonr(nTrain['nitrateMg'], nTrain['conductivity'])
# print("Conductivity and nitrate")
# print("DWT distance normalised: " + str(dwtD/2))
# print("Preason cc: " + str(pcc[0]))

# print("-----")
# dwtD = get_dtw_d(nTrain['nitrateMg'], nTrain['n'])
# pcc = pearsonr(nTrain['nitrateMg'], nTrain['n'])
# print("Nitrate and nitrate")
# print("DWT distance normalised: " + str(dwtD/2))
# print("Preason cc: " + str(pcc[0]))

''' Other robust regressors '''
# xTrain, xTest, X1, yTrain, X2, yTest = data_prep_n(X, y, lag)
# RANSAC_regresssion(xTrain, xTest, X1, yTrain, X2, yTest)
# Huber_regression(xTrain, xTest, X1, yTrain, X2, yTest)
# MLP_regression(xTrain, xTest, X1, yTrain, X2, yTest)

# plt.plot(nitrateMg, '-')
plt.show()

