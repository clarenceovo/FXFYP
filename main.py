import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels
from talib import MA_Type
def getdata():
    global tick
    global cftc
    tick = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/USDJPYH4(2016-2017).csv',usecols=[0,1,2,3,4,5]) # Tick Data H4, from 2016-01-01 to 2017-12-31
    cftc= pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/CFTC-097741_FO_ALL_CR.csv',usecols=[0,1,2,3,4,5,6,]) # CTFC Report on Future and Option Position Data
    #print(cftc)
    tick=tick.drop_duplicates(keep=False)
    #get date ,time ,open ,high ,low and close data

def candle_baranalysis():
    opprice=tick['Open'].astype(float).values
    hiprice=tick['High'].astype(float).values
    lowprice=tick['Low'].astype(float).values
    closeprice=tick['Close'].astype(float).values
    #create different array for pattern analusis using TA lib
    bollupper,bollmiddle,bolllower = talib.BBANDS(closeprice,matype=MA_Type.T3) #make BB Band using Close Price


def dataparsing():
    """
        1. Normalize the data (optional)
        2. Split the data to test & train dataset (70:30)

        """
    trainX = tick[:int(0.7 * len(tick))]  # 70% train ,30% Test
    testX = tick[int(0.7 * len(tick)):]
    trainY = tick[:int(0.7 * len(tick))]
    testY = tick[int(0.7 * len(tick)):]

def learning():
    """
    1. Build Seq model
    2. Use LSTM cell / ReLU cell (both will be tested)
    3. Set learning rate
    4. Record the RMSE score
        """
    return 0

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
    2. Prepare Data
    3.Create Model 
    4. Model evaluation and data visualisation 
    """
    getdata()
    candle_baranalysis()
    learning()
