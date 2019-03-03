from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.barfeed import csvfeed , yahoofeed
from pyalgotrade.technical import cross
from sklearn.externals import joblib
import talib

import pyalgotrade
demo_cash =100000

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
    def onBars(self, bars): #amend and implement the strategy in this function
        bar = bars[self.__instrument]
        self.info(bar.getClose())
        if self.__longPos is None: #if no long position
            shares = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice())*10000))
            self.__longPos=self.enterLong(self.__instrument,shares,True)

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

if __name__=='__main__':
    global model
    model=joblib.load('model.sav')
    run_strategy()