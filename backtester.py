import pandas as pd
import numpy as np
from datetime import datetime
from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.barfeed import csvfeed , yahoofeed
from pyalgotrade.technical import cross

import pyalgotrade


class machine_learning_strategy(strategy.BacktestingStrategy):

    def __init__(self, feed,instrument ): #initializing order
        super(machine_learning_strategy, self).__init__(feed)
        # We want a 15 period SMA over the closing prices.
        self.__instrument=instrument
        self.__longPos = None
        self.__shortPos = None

    def onExitOk(self, position):
        if self.__longPos == position:
            self.__longPos = None
        elif self.__shortPos == position:
            self.__shortPos = None
        else:
            assert (False)
    def onBars(self, bars):
        bar = bars[self.__instrument]
        self.info(bar.getClose())
    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        position.exitMarket()


    def onEnterCanceled(self, position): #cancel order before execution
        self.__position = None



    def onExitOk(self, position): #exit position
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

def run_strategy():
    # Load the bar feed from the CSV file
    feed = csvfeed.GenericBarFeed(frequency=pyalgotrade.barfeed.Frequency.HOUR,timezone=None, maxLen=1024)
    feed.addBarsFromCSV(instrument="gbpusd",path="C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/GBPUSD60.csv") #FIXED BARFEED ISSUE
    myStrategy = machine_learning_strategy(feed, "gbpusd")
    myStrategy.run()
   # print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())

run_strategy()