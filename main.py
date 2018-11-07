import pandas as pd
import numpy as np
import dateutil.parser
import matplotlib.pyplot as plt
#import talib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from talib import MA_Type

import math
def getdata():
    global tick
    global cftc
    tick = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/USDJPYH4(2016-2017).csv',usecols=[0,1,2,3,4,5]) # Tick Data H4, from 2016-01-01 to 2017-12-31
    cftc= pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/CFTC-097741_FO_ALL_CR.csv',usecols=[0,1,2,3,4,5,6,]) # CTFC Report on Future and Option Position Data
    #print(cftc)
    tick=tick.drop_duplicates(keep=False)
    print('INFO:Tick data is extracted successfully')
    #get date ,time ,open ,high ,low and close data

def candle_baranalysis():
    global opprice , hiprice , lowprice ,closeprice , dataset, bollupper,bollmiddle,bolllower
    opprice=tick['Open'].astype(float).values
    hiprice=tick['High'].astype(float).values
    lowprice=tick['Low'].astype(float).values
    closeprice=tick['Close'].astype(float).values
    #create different array for pattern analusis using TA lib
    #bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,matype=MA_Type.T3) #make BB Band using Close Price
    print('INFO:Candle Parameters is created')
def technical_analysis():

   # global macd
    #macd = talib.MACD(opprice) #MACD
    ##print (macd)
    print ('INFO: Technical Parameters is calculated')
def traceback(data,look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX), np.array(dataY)

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)



def parsing_learning():
    global trainX ,trainY , testX, testY ,look_back ,trainPredict , testPredict, scaler ,benchmark
    time = tick['Date'] +' '+tick['Time'] #merge date and time
    time = time.apply (lambda x :dateutil.parser.parse(x)) #Parse the time to fit the transform
    time=time.values
    #dataset = pd.DataFrame(data=[opprice, hiprice, lowprice, closeprice]).transpose()
    #dataset.columns=['Open','High','Low','Close']
    dataset = pd.DataFrame(data=opprice)
    dataset.dropna(inplace=True) # move empty dataset
    benchmark = dataset.values #dataset
    n = benchmark.shape[0]
    p = benchmark.shape[1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    benchmark = scaler.fit_transform(benchmark)
    train_size = int(len(benchmark) * 0.67)
    test_size = len(benchmark) - train_size
    train = benchmark[0:train_size, :]
    test=benchmark[train_size:len(benchmark), :]
    look_back = 6 #look back 6 hours
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=100, verbose=2)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
##Main Loop##

def data_visual():

    trainPredictPlot = np.empty_like(benchmark)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(benchmark)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(benchmark) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(benchmark))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
if __name__ == '__main__':
    """
    1. Read Data
    2. Prepare Data and talib data set creation 
    3. Create Model 
    4. Model evaluation and data visualisation 
    """
    getdata()
    candle_baranalysis()
   # technical_analysis()
    parsing_learning()
    data_visual()