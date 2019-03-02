import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error , jaccard_similarity_score
from keras.models import Sequential
from keras.layers import Dense ,Dropout ,Activation ,Input , TimeDistributed
from keras.layers import LSTM
from datetime import datetime
from talib import MA_Type
import talib
import math





def getdata():
    global tick ,cftc , us_home_sale, us_nonfarm_payroll
    tick = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/GBPUSD60.csv',usecols=[0,1,2,3,4,5],skiprows=1)
    tick.columns = ['Date','Open','High','Low','Close','Volumn']
    tick['Date']=tick['Date'].apply(lambda y: datetime.strptime(y,'%Y-%m-%d %H:%M:%S'))
    tick=tick[(tick['Date'].dt.year >=2012)] #data range(2012-2018)
    tick['Date'] = tick['Date'].apply(lambda y: datetime.strftime(y,"%Y-%m-%d %H:%M:%S"))
    tick=tick.drop_duplicates(keep=False)
    print('INFO:All data is extracted successfully')

def candle_baranalysis():
    global opprice , hiprice , lowprice ,closeprice , dataset, bollupper,bollmiddle,bolllower , avg_price,date
    date =tick['Date'].values
    opprice=tick['Open'].astype(float).values
    hiprice=tick['High'].astype(float).values
    lowprice=tick['Low'].astype(float).values
    closeprice=tick['Close'].astype(float).values
    avg_price = talib.AVGPRICE(opprice, hiprice, lowprice, closeprice)
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
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def parsing_learning():
    global trainX ,trainY , testX, testY ,look_back ,trainPredict , testPredict, scaler ,dataset ,epoch,batch ,dropout ,history ,testset ,eva_set
    look_back =1
    dataset = pd.DataFrame(data=[avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR]).transpose()
    dataset.columns = ['avg_price', 'SMA240', 'BBupper', 'BBMiddle' ,'BBLower', 'ATR']
    eva_set = dataset
    dataset.dropna(inplace=True)
    #print(eva_set)
    data = dataset.values
    data = data.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data=series_to_supervised(data,1,1)
    data.drop(data.columns[[7,8, 9, 10, 11]], axis=1, inplace=True)
    print(data.head(5))
    data = data.values #DF to list
    train_size = int(len(data)*0.7) #70% train ,30% test
    trainset = data[:train_size, :]
    testset = data[train_size:, :]
    trainX, trainY = trainset[:, :-1], trainset[:, -1]
    testX, testY = testset[:, :-1], testset[:, -1]
    testset = testY
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    epoch = 200
    batch=1000
    dropout=0.3
    model.add(Dense(48, kernel_initializer='normal', activation='sigmoid',input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(64,recurrent_activation=
                   'sigmoid', return_sequences=True)) #need stacked LSTM network? Eg Two LSTM layers
    model.add(Dropout(dropout))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(16))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
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
    testScore = math.sqrt(mean_squared_error(testY, testPredict))

def data_visual():
    global export_result
    trainPredictPlot = np.empty((avg_price.shape[0],1))
    export_result =  np.empty((avg_price.shape[0],1))
    avg = np.empty((avg_price.shape[0]))
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back+239:len(trainPredict) + look_back+239,:] = trainPredict.reshape((trainPredict.shape[0],1))
    export_result[look_back+239:len(trainPredict) + look_back+239,:] =trainPredict.reshape((trainPredict.shape[0],1))
    testPredictPlot = np.empty((avg_price.shape[0],1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + look_back +239 :len(avg_price) ,:] = testPredict.reshape((testPredict.shape[0],1))
    export_result[len(trainPredict) + look_back +239 :len(avg_price) ,:]= testPredict.reshape((testPredict.shape[0],1))
    export_result = export_result.ravel()
    test=eva_set['avg_price'].values
    avg = avg_price
    export_df = pd.DataFrame([date,avg, export_result, sma_240, ATR, bollupper, bollmiddle, bolllower])
    export_df=export_df.transpose()
    export_df.columns=['Date','Actual','Prediction','SMA','ATR','BBUpper','BBMiddle','BBLower']
    export_df.dropna(inplace=True)
    export_df.to_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/Dataset_toDashboard.csv')
    plt.plot(avg_price,label='Actual Price')
    plt.plot(trainPredictPlot,label='Train Prediction')
    plt.plot(testPredictPlot,label='Test Prediction')
    plt.title('GBPUSD H1 Prediction(Multivariate)')
    plt.legend()
    #plt.savefig(f'C:/Users/LokFung/Desktop/IERGYr4/IEFYP/PLT_IMAGE/GBPUSD60.png')
    #plt.show()

def model_eva(): #abs ERROR CALCULATION
    test = eva_set['avg_price'].values
    test = test[len(test) - len(testPredict):]
    score = mean_absolute_error(test,testPredict)
    mse = mean_squared_error(test,testPredict)
    #MAPE
    test , testpredict = np.array(test) , np.array(testPredict)
    mape = np.mean(np.abs((test-testpredict)/test))*100
    #jac = jaccard_similarity_score(test,testPredict)
    print (f'Mean Absolute Error Score: {score}')
    print(f'Mean Square Error Score: {mse}')
    print (f'MAPE :{mape}')


if __name__ == '__main__':
    getdata()
    candle_baranalysis()
    technical_analysis()
    parsing_learning()
    model_eva()
    #data_visual()
