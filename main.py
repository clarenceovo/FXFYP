import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense ,Dropout ,Activation ,Input , TimeDistributed
from keras.layers import LSTM
from datetime import datetime
from talib import MA_Type
import talib
import math
from scipy.misc import derivative
def getdata():
    global tick ,cftc , us_home_sale, us_nonfarm_payroll
    tick = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/AUDUSD60.csv',usecols=[0,1,2,3,4,5],skiprows=1)
    tick.columns = ['Date','Open','High','Low','Close','Volumn']
    tick['Date']=tick['Date'].apply(lambda y: datetime.strptime(y,'%d.%m.%Y %H:%M:%S.000'))
    tick=tick[(tick['Date'].dt.year >=2015)] #data range(2011-2018)
    #print(tick)
    """
    We are not deal with multiple time lag problem in this stage
    #cftc= pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/CFTC-097741_FO_ALL_CR.csv',usecols=[0,5,6]) # CTFC Report on Future and Option Position Data ,2nd feature
    #print(cftc)
    #us_home_sale = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/US_NEW_HOME_SALE(2016-2017).csv',usecols=[0,1]) #3rd param
    #us_home_sale['Date'] = us_home_sale['Date'].apply(lambda y: datetime.strptime(y,'%d-%b-%y')) #convert to datetime object
    us_home_sale['Actual'] = us_home_sale['Actual'].apply( lambda y: int(y.strip('K')) * 1000)  # conver to K to int object
    us_home_sale['Actual'] = us_home_sale['Actual'].apply(lambda y: int(y.strip('K'))*1000) #conver to K to int object
    #us_nonfarm_payroll = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/US_NONFARM_PAYROLL(2016-2017).csv',usecols=[0,1]) #4th param
    #us_nonfarm_payroll['Date'] = us_nonfarm_payroll['Date'].apply(lambda y: datetime.strptime(y, '%d-%b-%y')) #convert to datetime object
    #us_nonfarm_payroll['Actual'] = us_nonfarm_payroll['Actual'].apply(lambda y: int(y.strip('K')) * 1000)
    """
    tick=tick.drop_duplicates(keep=False)
    print('INFO:All data is extracted successfully')

def candle_baranalysis():
    global opprice , hiprice , lowprice ,closeprice , dataset, bollupper,bollmiddle,bolllower , avg_price
    opprice=tick['Open'].astype(float).values
    hiprice=tick['High'].astype(float).values
    lowprice=tick['Low'].astype(float).values
    closeprice=tick['Close'].astype(float).values
    avg_price = talib.AVGPRICE(opprice, hiprice, lowprice, closeprice)
    #create different array for pattern analusis using TA lib
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,nbdevup=2, nbdevdn=2,matype=MA_Type.T3) #make BB Band using Close Price
    print('INFO:Candle Parameters is created')
def technical_analysis():
    global sma_5, sma_24, sma_50 ,sma_240 , ATR ,RSI
    sma_240 = talib.SMA(opprice, timeperiod=240) #sma_240 = 10 Days Avg
    ATR =talib.ATR(hiprice,lowprice,closeprice,timeperiod=24)
    RSI = talib.RSI(closeprice,120)
    print ('INFO: Technical Parameters is calculated')
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
    global trainX ,trainY , testX, testY ,look_back ,trainPredict , testPredict, scaler ,dataset ,epoch,batch ,dropout ,history
    look_back =1
    dataset = pd.DataFrame(data=[avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR]).transpose()
    dataset.columns = ['avg_price', 'SMA240', 'BBupper', 'BBMiddle' ,'BBLower', 'ATR']  # NAMING COLUMN
    dataset.dropna(inplace=True)
    data = dataset.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data=series_to_supervised(data,1,1)
    data.drop(data.columns[[7,8, 9, 10, 11]], axis=1, inplace=True)
    data = data.values #DF to list
    train_size = int(len(data)*0.7) #70% train ,30% test
    trainset = data[:train_size, :]
    testset = data[train_size:, :]
    trainX, trainY = trainset[:, :-1], trainset[:, -1]
    testX, testY = testset[:, :-1], testset[:, -1]
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    epoch=200
    batch=1000
    dropout=0.6
    model.add(Dense(48, kernel_initializer='normal', activation='relu',input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(64,recurrent_activation=
                   'relu', return_sequences=True)) #need stacked LSTM network? Eg Two LSTM layers
    model.add(Dropout(dropout))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(16))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse', 'mae', 'mape'])
    history =model.fit(trainX, trainY, epochs=epoch, batch_size=batch, verbose=2,shuffle=True)
    trainPredict = model.predict(trainX)
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))  # reshape the data
    trainresult = np.concatenate((trainPredict, trainX[:, 1:]), axis=1)
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    #calculate the SD SCORE
    trainresult = scaler.inverse_transform(trainresult)
    trainPredict = trainresult[:, 0]
    testPredict = model.predict(testX)
    testX = testX.reshape((testX.shape[0], testX.shape[2]))  #reshape the data
    testresult = np.concatenate((testPredict, testX[:, 1:]), axis=1)
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    testresult  = scaler.inverse_transform(testresult)
    testPredict = testresult[:,0]
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    #"""
def data_visual():
    trainPredictPlot = np.empty((avg_price.shape[0],1))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back+239:len(trainPredict) + look_back+239,:] = trainPredict.reshape((trainPredict.shape[0],1))
    testPredictPlot = np.empty((avg_price.shape[0],1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + look_back +239 :len(avg_price) ,:] = testPredict.reshape((testPredict.shape[0],1))
    plt.plot(avg_price,label='AUDUSD Actual Price')
    plt.plot(trainPredictPlot,label='Train Prediction')
    plt.plot(testPredictPlot,label='Test Prediction')
    plt.title('AUDUSD H1 Prediction(Multivariate)')
    #plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
    #plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
    #plt.plot(history.history['mean_absolute_percentage_error'], label='mean_squared_error')
    #plt.title('Model Performance')
    plt.legend()
    plt.savefig(f'C:/Users/LokFung/Desktop/IERGYr4/IEFYP/PLT_IMAGE/GBPUSD60.png')
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