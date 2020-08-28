from loadData import *
from tools import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import *
import pywt

index, time, conductivity, turbidity, nitrateMg, nitrateUm = load_c_data()

'''
2 methods: 
    1:  regression as best as possible without wavelet first 
        get r2
    2:  wavelet decompose into two waves
        regression for each done seperately
        combine to give final model
        get r2
    3: same as two but instead of regressing noise, get zero vector

may need to move regression details here to do a better regression

Data:
    Use only one seasons worth of data
    This could be cleaned up to avoid repeating calculations
'''

''' isolate season data '''
data = pandas.DataFrame({'time': time, 'nitrateMg' : nitrateMg, 'index' : index})
data.set_index('time', inplace=True)

dat = data
data = pandas.DataFrame({'nitrateMg': dat['nitrateMg']})

''' Method 1 '''
r=[]
names=[]
lag = 112 # lag window
# for 0 to lag window, shift by one then add
y = data['nitrateMg'].shift(lag)
y = y[lag:]
x = apply_lag(data, lag)
x = x[lag:]

# train test split 
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.50, shuffle=False)
x1 = yTrain.index.values
x2 = yTest.index.values

scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

scalerY= StandardScaler()
scalerY.fit(np.array(yTrain).reshape(-1, 1))
yTrain = scalerY.transform(np.array(yTrain).reshape(-1, 1))
yTest = scalerY.transform(np.array(yTest).reshape(-1, 1))

mlpModel = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=4, 
                        alpha=20, max_iter=5000, n_iter_no_change=10)
mlpReg = mlpModel.fit(xTrain, np.array(yTrain).ravel())
mlpTrainPrediction = mlpReg.predict(xTrain)
mlpTestPrediction = mlpReg.predict(xTest)

r.append([mlpTrainPrediction, mlpTestPrediction, 
        mean_absolute_error(yTest, mlpTestPrediction), 
        mlpModel.score(xTest, yTest)])
names.append('mlp scaled')

mlpR2 = mlpModel.score(xTest, yTest)

print("M1: R2: " + str(mlpR2) + " MSE " + 
        str(mean_squared_error(yTest, mlpTestPrediction)) + "\n")

residuals = abs(mlpTrainPrediction - yTrain)
mean = np.mean(residuals)
stdev = np.std(residuals)
threshold = mean + 0.5*stdev

mask, outlierCountTest = get_mask_from_threshold(yTest, mlpTestPrediction, threshold)
maskedYTest, outliersYTest = apply_mask_outlier(yTest, mask)
mask, outlierCountTrain = get_mask_from_threshold(yTrain, mlpTrainPrediction, threshold)
maskedYTrain, outliersYTrain = apply_mask_outlier(yTrain, mask)

outlierCount = outlierCountTest + outlierCountTrain

yTrain = scalerY.inverse_transform(np.array(yTrain).reshape(-1, 1))
yTest = scalerY.inverse_transform(np.array(yTest).reshape(-1, 1))
maskedYTest = scalerY.inverse_transform(np.array(maskedYTest).reshape(-1, 1))
maskedYTrain = scalerY.inverse_transform(np.array(maskedYTrain).reshape(-1, 1))
outliersYTest = scalerY.inverse_transform(np.array(outliersYTest).reshape(-1, 1))
outliersYTrain = scalerY.inverse_transform(np.array(outliersYTrain).reshape(-1, 1))

plt.figure()
plt.suptitle("Outlier detection using prediction, found %d outliers" % outlierCount)
plt.scatter(x2, outliersYTest, s=1, c='r', alpha=0.2)
plt.plot_date(x2, maskedYTest, '-', c='k', alpha=0.8)
plt.scatter(x1, outliersYTrain, s=1, c='r', alpha=0.2)
plt.plot_date(x1, maskedYTrain, '-', c='k', alpha=0.8)


''' Method 3 '''
data = pandas.DataFrame({'nitrateMg': dat['nitrateMg'], 'index':dat['index']})
lag = 10
rollingNitrate = data['nitrateMg'].rolling(lag).mean().values
data['rollingNitrate'] = rollingNitrate
# Chose not to include rolling mean for now

n = 3
coef =  pywt.wavedec(data['nitrateMg'], 'db1', level=n)
original = pywt.waverec(coef, 'db1')

coef[1] = np.zeros_like(coef[1]) # replace high frequency with 0
coef[2] = np.zeros_like(coef[2])

# low frequency signal by removing high freqneucy before reconstruction
lowY = pywt.waverec(coef, 'db1')
lowY = pandas.DataFrame({'lowY': lowY})
lag = 100 # lag window

y = lowY.shift(lag)
x = apply_lag(lowY, lag)
y = lowY[lag:]
x = x[lag:]

# train test split 
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.50, shuffle=False)
x1 = yTrain.index.values
x2 = yTest.index.values
originalY = original[lag:]
print(len(yTrain), len(originalY[0:len(originalY)/2]))
# scaling 
scaler = StandardScaler()
scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

scalerY= StandardScaler()
scalerY.fit(np.array(yTrain).reshape(-1, 1))
yTrain = scalerY.transform(np.array(yTrain).reshape(-1, 1))
yTest = scalerY.transform(np.array(yTest).reshape(-1, 1))
y = scalerY.transform(np.array(y).reshape(-1, 1))
original = scalerY.transform(np.array(original).reshape(-1, 1))
lowY = scalerY.transform(np.array(lowY).reshape(-1, 1))
originalY = scalerY.transform(np.array(originalY).reshape(-1, 1))

mlpModel = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=4, 
                        alpha=20, max_iter=5000, n_iter_no_change=10)
mlpReg = mlpModel.fit(xTrain, np.array(yTrain).ravel())
mlpTrainPrediction = mlpReg.predict(xTrain)
mlpTestPrediction = mlpReg.predict(xTest)
mlpR2 = mlpModel.score(xTest, yTest)

print("M3 R2: " + str(mlpR2) + " MSE " + 
        str(mean_squared_error(original[original.size/2+50:], mlpTestPrediction))
        + "\n")


# caluculate residuals and calssify outliers
yTrain = np.array(yTrain)
yTest = np.array(yTest)
residuals = abs(mlpTrainPrediction - originalY[0:len(originalY)/2])
mean = np.mean(residuals)
stdev = np.std(residuals)
threshold = mean + 0*stdev

residuals = abs(mlpTrainPrediction - yTrain)
mean = np.mean(residuals)
stdev = np.std(residuals)
thresholdOld = mean + 0*stdev
print(threshold, thresholdOld)
threshold = thresholdOld

mask, outlierCountTest = get_mask_from_threshold(originalY[len(originalY)/2:], mlpTestPrediction, threshold)
maskedYTest, outliersYTest = apply_mask_outlier(originalY[len(originalY)/2:], mask)
mask, outlierCountTrain = get_mask_from_threshold(originalY[0:len(originalY)/2], mlpTrainPrediction, threshold)
maskedYTrain, outliersYTrain = apply_mask_outlier(originalY[0:len(originalY)/2], mask)

outlierCount = outlierCountTest + outlierCountTrain

outliersYTest = scalerY.inverse_transform(np.array(outliersYTest).reshape(-1, 1))
outliersYTrain = scalerY.inverse_transform(np.array(outliersYTrain).reshape(-1, 1))
maskedYTest = scalerY.inverse_transform(np.array(maskedYTest).reshape(-1, 1))
original = scalerY.inverse_transform(np.array(original).reshape(-1, 1))
maskedYTrain = scalerY.inverse_transform(np.array(maskedYTrain).reshape(-1, 1))
lowY = scalerY.inverse_transform(np.array(lowY).reshape(-1, 1))
mlpTrainPrediction = scalerY.inverse_transform(np.array(mlpTrainPrediction).reshape(-1, 1))
mlpTestPrediction = scalerY.inverse_transform(np.array(mlpTestPrediction).reshape(-1, 1))

plt.figure()
plt.suptitle("Outlier detection using prediction after decomposition, found %d outliers" % outlierCount)
plt.plot(original[0:100], '.', markersize=2, color='r', alpha=0.2)
plt.plot(x2, outliersYTest, '.', markersize=2, color='r', alpha=0.2)
plt.plot(x1, outliersYTrain, '.', markersize=2, color='r', alpha=0.2)
plt.plot(x2, maskedYTest, '-', c='k', alpha=0.8)
plt.plot(x1, maskedYTrain, '-', c='k', alpha=0.8)

# comparison of original signal and low frequency signal from decomp
plt.figure()
plt.plot(original, '.', c='k', markersize=3, label='Original signal')
plt.plot(lowY, '.', c = 'C1', markersize=2, label='Low frequency after decomposition')
plt.legend()

#plot of prediciton
plt.figure()
plt.plot(lowY, '.', markersize=3, c='k')
plt.plot(x1, mlpTrainPrediction, c='C0')
plt.plot(x2, mlpTestPrediction, c='C0')

# plots of original signal, low frequency signal, and outlier removed signals 
#   for comparison
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(original, '-', c='0.7', markersize=2)
plt.title('Original signal')
plt.subplot(3, 1, 2)
plt.plot(lowY, '-', c = 'C4', markersize=2)
plt.title('Low frequency after decomposition')
plt.subplot(3, 1, 3)
plt.plot(x2, maskedYTest, '-', c = 'C4', markersize=2)
plt.plot(x1, maskedYTrain, '-', c = 'C4', markersize=2)
plt.title("Decomposed and without outliers")


plt.show()