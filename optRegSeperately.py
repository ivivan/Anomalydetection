# optimising and visualising regression methods, eliminated gaps

import matplotlib.pyplot as plt
import numpy as np
import pandas
from loadData import *
# from statMethods import *
from tools import *
# from clusteringMethods import *
# from regressionMethods import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
from sklearn import linear_model 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse

index, time, conductivity, turbidity, nitrateMg, nitrateUm = load_c_data()

dat = pandas.DataFrame({'time': time, 'nitrateMg' : nitrateMg, 'index' : index, 
                        'conductivity': conductivity, 'turbidity' : turbidity})


# add the rolling mean of nitrate values as a column
window = 10
rollingNitrate = dat['nitrateMg'].rolling(window).mean().values
dat['rollingNitrate'] = rollingNitrate

dat = dat[10:] # first 10 have a rolling mean of NaN and so must be removed 

### D ###
beforeTimes = ["2016-12-08", "2017-7-01", "2018-1-01", "2018-6-02"]
afterTimes = ["2017-7-01", "2018-1-01", "2018-6-02", "2018-08-24"]

# hold data for display later
r = []
names = []
x1s = []
x2s = []
yTrains = []
yTests = []
outlierCount = 0

# preprocessing:
lag = 112 # lag window
# features included in the model
incFeatures = pandas.DataFrame({'time':dat['time'], 
                                'nitrateMg':dat['nitrateMg'], 
                                'rollingNitrate':dat['rollingNitrate']})
dat.set_index('time', inplace=True)
incFeatures.set_index('time', inplace=True)

y = incFeatures['nitrateMg'].shift(lag)
y = y[lag:]
x = apply_lag(incFeatures, lag)
x = x[lag:]

formattedData = pandas.DataFrame({'y': y})
# not really supposed to do this but convenient
formattedData['x'] = sparse.coo_matrix(x).toarray().tolist() 

for i in range(0, len(beforeTimes)):
    data = formattedData.truncate(before=datetime.strptime(beforeTimes[i], '%Y-%m-%d'), 
                        after=datetime.strptime(afterTimes[i], '%Y-%m-%d'))
    
    
    y = data['y']
    x = []
    for a in data['x'].values:
        x.append(np.array(a))

    # train test split 
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.50, shuffle=False)

    # independant axis values for visualisation
    x1 = yTrain.index.values
    x2 = yTest.index.values

    # scaling in both x and y
    scaler = StandardScaler()
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    scalerY= StandardScaler()
    scalerY.fit(np.array(yTrain).reshape(-1, 1))
    yTrain = scalerY.transform(np.array(yTrain).reshape(-1, 1))
    yTest = scalerY.transform(np.array(yTest).reshape(-1, 1))

    # model and prediction
    mlpModel = MLPRegressor(random_state=4, hidden_layer_sizes=(10, 10, 10), 
                            alpha=20, max_iter=5000, n_iter_no_change=10)
    mlpReg = mlpModel.fit(xTrain, np.array(yTrain).ravel())
    mlpTrainPrediction = mlpReg.predict(xTrain)
    mlpTestPrediction = mlpReg.predict(xTest)
    
    # scale predictions to normal scale before storing
    r.append([scalerY.inverse_transform(mlpTrainPrediction), 
              scalerY.inverse_transform(mlpTestPrediction), 
              mean_absolute_error(yTest, mlpTestPrediction), 
              mlpModel.score(xTest, yTest)])
    names.append('Range %d' % i)

    ''' THRESHOLDS '''
    residuals = abs(mlpTrainPrediction - yTrain)
    mean = np.mean(residuals)
    stdev = np.std(residuals)
    threshold = mean + 0.2*stdev

    # array of just inflier values based on threshold
    maskTest, outlierCountTest = get_mask_from_threshold(yTest, mlpTestPrediction, threshold)
    maskTrain, outlierCountTrain = get_mask_from_threshold(yTrain, mlpTrainPrediction, threshold)

    # convert back so proper scale
    yTrain = scalerY.inverse_transform(np.array(yTrain).reshape(-1, 1))
    yTest = scalerY.inverse_transform(np.array(yTest).reshape(-1, 1))
    
    # apply mask to proper scale values
    maskedYTest, outliersYTest = apply_mask_outlier(yTest, maskTest)
    maskedYTrain, outliersYTrain = apply_mask_outlier(yTrain, maskTrain)

    outlierCount += outlierCountTest + outlierCountTrain

    yTrains.append(yTrain)
    yTests.append(yTest)
    x1s.append(x1)
    x2s.append(x2)

    # Add region to the main plots
    fig1 = plt.figure(1)
    plt.suptitle("Nitrate data with outliers marked. Outliers: %d (%d)" % (outlierCount, outlierCount))
    out = plt.scatter(x2, outliersYTest, s=1, c='r')
    train = plt.scatter(x2, maskedYTest, s=1, c='C0')
    plt.scatter(x1, outliersYTrain, s=1, c='r')
    test = plt.scatter(x1, maskedYTrain, s=1, c='k')

    plt.figure(2)
    plt.title("Outlier detection using prediction, found %d outliers" % outlierCount)
    plt.scatter(x2, outliersYTest, s=2, c='r', alpha=0.2)
    plt.plot_date(x2, maskedYTest, '-', c='k', alpha=0.8)
    plt.scatter(x1, outliersYTrain, s=2, c='r', alpha=0.2)
    plt.plot_date(x1, maskedYTrain, '-', c='k', alpha=0.8, label='Outliers %d' % outlierCount)
    plt.axvspan(x1[0], x1[-1], color='0.8', alpha=0.2)
    plt.ylabel("Nitrate levels (mg/L)")
    
fig1.legend((out, train, test), ('Outliers', 'Training region', 'Testing region'))

# plot each region seperately
fig = plt.figure()
plt.suptitle("Model on each time interval")
for i in range(0, len(names)):
    plots = len(names)
    plt.subplot(plots, 1, i+1)
    a1 = plt.scatter(x1s[i], yTrains[i], c='0.5', s=2,  alpha=0.5)
    r1 = plt.plot(x1s[i], r[i][0], color='0.1')
    a2 = plt.scatter(x2s[i], yTests[i], s=2, c = '0.5', alpha=0.5)
    r2 = plt.plot(x2s[i], r[i][1], color='C1')
    plt.title(names[i])
    plt.ylabel('nitrate levels (mg/L)')
plt.figlegend((a1, r1[0], r2[0]), ('True values', 
                                'Training set model values', 
                                'Testing set model values'))
plt.show()
