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

def parsing_learning():
    global trainX ,trainY , testX, testY
    time = tick['Date'] +' '+tick['Time'] #merge date and time
    time = time.apply (lambda x :dateutil.parser.parse(x)) #Parse the time to fit the transform
    time=time.values
    dataset = pd.DataFrame(data=[opprice, hiprice, lowprice, closeprice]).transpose()
    dataset.columns=['Open','High','Low','Close']
    dataset.dropna(inplace=True) # move NA dataset
    #print(dataset)
    benchmark = dataset['Open'].values
    #benchmark = benchmark.astype('float')
    print (benchmark)
    #testsize = len(benchmark)-trainsize
    #scaler =MinMaxScaler(0,1)
    #set = scaler.fit_transform(benchmark)
    trainsize = len(benchmark) * 0.7
    train = set[0:trainsize]
    test = set[trainsize:len(benchmark)]
    trainX,trainY = traceback(train,1)
    testX , testY = traceback(test,1)
    #print(trainX)
    #print(trainY) #t+1
    trainX  = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
    testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
    """
    values = dataset.values
    dataencoder = LabelEncoder()
    values[:,6] = dataencoder.fit_transform(values[:,6])
    datascaler = MinMaxScaler(feature_range=(0,1))
    parsed = datascaler.fit_transform(values)
    """
    #train = parsed[:int(0.7 * len(parsed)),:]  # 70% train ,30% Test
    #test = parsed[int(0.7 * len(parsed)):,:]
    #trainX ,trainY = train[:,:-1],train[:,-1]
    #testX, testY = test[:, :-1], test[:, -1]
    #trainX=trainX.reshape((trainX.shape[0],1,trainX.shape[1]))
    #testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    print('INFO:The dataset is parsed')
    global result , prediction
    model = Sequential()
    model.add(LSTM(4,input_shape=(1,1)))
    model.add(Dense(1))# Add one dense layer
    model.compile(loss='mse',optimizer='rmsprop')
    model.fit(trainX,trainY,epochs=100,batch_size=100,verbose=2 )
    trainprediction = model.predict(testX)
    testprediction = model.predict(testX)
    #trainprediction = scaler.inverse_transform(trainprediction)
    #trainY=scaler.inverse_transform([trainY])
    #testprediction=scaler.inverse_transform(testprediction)
    #testY=scaler.inverse_transform([testY])
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainprediction[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testprediction[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
##Main Loop##
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