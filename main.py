import pandas as pd
import numpy as np
import dateutil.parser
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
from talib import MA_Type
import talib
def getdata():
    global tick ,cftc , us_home_sale, us_nonfarm_payroll
    tick = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/USDJPYH4(2016-2017).csv',usecols=[0,1,2,3,4,5]) # Tick Data H4, from 2016-01-01 to 2017-12-31
    """
    We are not deal with multiple time lag problem in this stage
    #cftc= pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/CFTC-097741_FO_ALL_CR.csv',usecols=[0,5,6]) # CTFC Report on Future and Option Position Data ,2nd feature
    #print(cftc)
    #us_home_sale = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/US_NEW_HOME_SALE(2016-2017).csv',usecols=[0,1]) #3rd param
    #us_home_sale['Date'] = us_home_sale['Date'].apply(lambda y: datetime.strptime(y,'%d-%b-%y')) #convert to datetime object
    #us_home_sale['Actual'] = us_home_sale['Actual'].apply(lambda y: int(y.strip('K'))*1000) #conver to K to int object
    #us_nonfarm_payroll = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/US_NONFARM_PAYROLL(2016-2017).csv',usecols=[0,1]) #4th param
    #us_nonfarm_payroll['Date'] = us_nonfarm_payroll['Date'].apply(lambda y: datetime.strptime(y, '%d-%b-%y')) #convert to datetime object
    #us_nonfarm_payroll['Actual'] = us_nonfarm_payroll['Actual'].apply(lambda y: int(y.strip('K')) * 1000)
    """
    tick=tick.drop_duplicates(keep=False)
    print('INFO:All data is extracted successfully')
def candle_baranalysis():
    global opprice , hiprice , lowprice ,closeprice , dataset, bollupper,bollmiddle,bolllower
    opprice=tick['Open'].astype(float).values
    hiprice=tick['High'].astype(float).values
    lowprice=tick['Low'].astype(float).values
    closeprice=tick['Close'].astype(float).values
    #create different array for pattern analusis using TA lib
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,nbdevup=2, nbdevdn=2,matype=MA_Type.T3) #make BB Band using Close Price
    print('INFO:Candle Parameters is created')
def technical_analysis():
    global sma_5, sma_20, sma_50 ,sma_120 , ATR_20 ,RSI_14
    sma_5 = talib.SMA(opprice, timeperiod=5)
    sma_20 = talib.SMA(opprice , timeperiod=20)
    #sma_50 = talib.SMA(opprice, timeperiod=50)
    #sma_120= talib.SMA(opprice, timeperiod=120)
    #add the the gradient of SMA
    ATR_20 =talib.ATR(hiprice,lowprice,closeprice,timeperiod=20) #todayATR = atr[-1]
    RSI_14 = talib.RSI(closeprice,14)
    #print(ATR_20)
    #print(RSI_14)
    print ('INFO: Technical Parameters is calculated')
"""
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
"""
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def parsing_learning():
    global trainX ,trainY , testX, testY ,look_back ,trainPredict , testPredict, scaler ,dataset
    look_back =1
    time = tick['Date'] +' '+tick['Time'] #merge date and time
    time = time.apply (lambda x :dateutil.parser.parse(x)) #Parse the time to fit the transform
    time=time.values
    dataset = pd.DataFrame(data=[opprice, sma_5,sma_20 ,bollupper,bolllower, ATR_20,RSI_14 ]).transpose()
    dataset.columns=['Price','SMA5','SMA20','BBupper','BBLower','ATR','RSI'] #NAMING COLUMN
    dataset.dropna(inplace=True)
    data = dataset.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data=series_to_supervised(data,1,1)
    data.drop(data.columns[[8,9, 10, 11, 12, 13 ]], axis=1, inplace=True)
    data = data.values #DF to list
    train_size = int(len(data)*0.7) #70% train ,30% test
    trainset = data[:train_size, :]
    testset = data[train_size:, :]
    trainX, trainY = trainset[:, :-1], trainset[:, -1]
    testX, testY = testset[:, :-1], testset[:, -1]
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(48, input_shape=(trainX.shape[1], trainX.shape[2]))) #need stacked LSTM network? Eg Two LSTM layers
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=50, verbose=2)
    trainPredict = model.predict(trainX)
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))  # reshape the data
    trainresult = np.concatenate((trainPredict, trainX[:, 1:]), axis=1)
    trainresult = scaler.inverse_transform(trainresult)
    trainPredict = trainresult[:, 0]

    testPredict = model.predict(testX)
    testX = testX.reshape((testX.shape[0], testX.shape[2]))  #reshape the data
    testresult = np.concatenate((testPredict, testX[:, 1:]), axis=1)
    testresult  = scaler.inverse_transform(testresult)
    testPredict = testresult[:,0]
    # calculate root mean squared error
    #trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    #print('Train Score: %.2f RMSE' % (trainScore))
    #testScore = math.sqrt(mean_squared_error(testY, testPredict))
    #print('Test Score: %.2f RMSE' % (testScore))

def data_visual():
    dataprice =dataset['Price'].values
    trainPredictPlot = np.empty((dataprice.shape[0],1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back,:] = trainPredict.reshape((trainPredict.shape[0],1))
    testPredictPlot = np.empty((dataprice.shape[0],1))
    testPredictPlot[:, :] = np.nan
    print(f'SHAPE OF Plot:{trainPredictPlot.shape}')
    print(f'SHAPE OF Train:{trainPredict.shape}')
    print(f'SHAPE OF TEST:{testPredict.shape}')
    testPredictPlot[len(trainPredict) + look_back  :,:] = testPredict.reshape((testPredict.shape[0],1))
    plt.plot(dataprice,label='USDJPY Actual Price')
    plt.plot(trainPredictPlot,label='Train Prediction')
    plt.plot(testPredictPlot,label='Test Prediction')
    #plt.plot(bollupper, label='Upper BB Band')
    #plt.plot(bolllower, label='Lower BB Band')
    plt.title('USDJPY H4 Prediction(Multivariate)')
    plt.legend()
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
    technical_analysis()
    parsing_learning()
    data_visual()