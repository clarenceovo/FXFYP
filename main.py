import pandas as pd
import numpy as np
import dateutil.parser
import matplotlib.pyplot as plt
import talib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from talib import MA_Type

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
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,matype=MA_Type.T3) #make BB Band using Close Price
    print('INFO:Candle Parameters is created')
def technical_analysis():
    """
     Create technical parameters using the tick data so that the model can put the param into account
    Types of Parameters will be calculated:
    1. MACD
    2.RSI
    3.
    """
    global macd
    macd = talib.MACD(opprice) #MACD
    #print (macd)
    print ('INFO: Technical Parameters is calculated')
def dataparsing():
    """
        1. Create data set
        2. Split the data to test & train dataset (70:30)

        """
    global trainX ,trainY , testX, testY
    time = tick['Date'] +' '+tick['Time'] #merge date and time
    time = time.apply (lambda x :dateutil.parser.parse(x)) #Parse the time to fit the transform
    time=time.values
    dataset = pd.DataFrame(data=[time,opprice, hiprice, lowprice, closeprice,bollupper,bollmiddle,bolllower]).transpose()
    print(dataset)
    dataset.columns=['Time','Open','High','Low','Close','BBUpper','BBMiddle','BBLower']
    dataset.dropna(inplace=True) # move NA dataset
    values = dataset.values
    dataencoder = LabelEncoder()
    values[:,7] = dataencoder.fit_transform(values[:,7])
    datascaler = MinMaxScaler(feature_range=(0,1))
    parsed = datascaler.fit_transform(values)
    print(parsed)
    trainX = parsed[:int(0.7 * len(parsed))]  # 70% train ,30% Test
    testX = parsed[int(0.7 * len(parsed)):]
    trainY = parsed[:int(0.7 * len(parsed))]
    testY = parsed[int(0.7 * len(parsed)):]

def learning():
  global result
  model = Sequential()
  model.add(LSTM(50,input_shape=(trainX.shape[1],trainX,shape[2]))) #50 Nodes
  model.add(Dense(1,activation='sigmoid')) # Add one dense layer
  model.compile(loss='mae',optimizer='rmsprop')
  supervised = model.fit(trainX,trainY,epochs=100,batch_size=20,validation_data=(testX,testY) ,verbose=2 ,shuffle=False)

def data_vis():
    """
    1. Plot the graph (actual and prediction)
    2. Save it to gif

    """
    return 0

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
    dataparsing()
    learning()
