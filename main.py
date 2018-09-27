import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels

def getdata():
    global df
    df = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/USDJPYH4(2016-2017).csv',usecols=[0,1,2,3,4,5])
    df=df.drop_duplicates(keep=False)
    #get date ,time ,open ,high ,low and close data

def candle_baranalysis():
    opprice=df['Open'].astype(float)
    hiprice=df['High'].astype(float)
    lowprice=df['Low'].astype(float)
    closeprice=df['Close'].astype(float)
    ##create different array for pattern analusis


def dataparsing():
    """
    1. Normalize the data (optional)
    2. Split the data to test & train dataset
    3.
    """

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
    getdata()
    candle_baranalysis()
    learning()