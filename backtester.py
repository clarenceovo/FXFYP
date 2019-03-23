from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.technical import macd
from pyalgotrade.technical import atr
from pyalgotrade.barfeed import csvfeed
from pyalgotrade.technical import cross
from sklearn.externals import joblib
from pyalgotrade.technical import bollinger
import talib
import pandas as pd
import pyalgotrade
demo_cash =100000
import numpy as np


class machine_learning_strategy(strategy.BacktestingStrategy):

    def __init__(self, feed,instrument ,initialCash,model): #initializing order
        super(machine_learning_strategy, self).__init__(feed,initialCash)
        # We want a 15 period SMA over the closing prices.
        SMA=14
        self.__instrument=instrument
        self.__longPos = None
        self.__shortPos = None
        #self.__macd =macd.MACD(feed[instrument].getPriceDataSeries())
        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(),240)
        self.__bbands=bollinger.BollingerBands(feed[instrument].getPriceDataSeries(),40,2)
        self.atr = atr.ATR(feed[instrument],24)

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
       # self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onExitOk(self, position):
        if self.__longPos == position:
            self.__longPos = None
            self.info('Long Position is canceled')

        elif self.__shortPos == position:
            self.__shortPos = None
            self.info('Short Position is canceled')
        else:
            assert (False)

    def onExitCanceled(self,position):
        # If the exit was canceled, re-submit it.
        if position is self.__longPos:
            self.__longPos.exitMarket()
        if position is self.__shortPos:
            self.__shortPos.exitMarket()

    def pos_check(self,position):
        return 0

    def onBars(self, bars): #amend and implement the strategy in this function
        bar = bars[self.__instrument]
        lower = self.__bbands.getLowerBand()[-1]
        middle = self.__bbands.getMiddleBand()[-1]
        upper = self.__bbands.getUpperBand()[-1]
        """
        Format of dataset in model training 
        [avg_price, sma_240, bollupper,bollmiddle, bolllower, ATR]
                
        """

        currentprice=round(bar.getPrice(),6)
        if self.__sma[-1] is None or self.__bbands.getLowerBand()[-1] is None or self.__bbands.getMiddleBand()[-1] is None or self.__bbands.getUpperBand()[-1] is None : #Wait to get enough info for tech indicator
            return
        lower = round(lower, 5) #BBlower
        middle = round(middle, 5) #BBLower
        upper = round(upper, 5) #BBUpper
        atr = self.atr[-1] #ATR
        dataset = np.array([round(bar.getClose(),5),self.__sma[-1], upper, middle, lower]) #recombine the array
        dataset = dataset.reshape((1, 1, -1))
        prediction  =model.predict(dataset) #

        print(prediction)
        if self.__longPos is None: #if no long position
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice())*100000)) #lot size is determined by the amount of cash we have in the demo

            if currentprice > round(self.__sma[-1],5): #round the 7 digits
                self.info(f'LONG Position ENTRY:{currentprice}')

                self.__longPos = self.enterLong(self.__instrument,lot_size*1000,True)

        elif currentprice < self.__sma[-1] and not self.__longPos.exitActive():  # EXIT RULE
            self.info(f'LONG Position EXIT:{currentprice}')
            self.__longPos.exitMarket()


        if self.__shortPos is None:
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice()) * 100000))
            if currentprice < round(self.__sma[-1],5):
                self.__shortPos = self.enterShort(self.__instrument, lot_size*1000, True)
                self.info(f'SHORT Position ENTRY:{currentprice}')

        elif currentprice > self.__sma[-1] and not self.__shortPos.exitActive():  # EXIT RULE
            self.info(f'SHORT Position EXIT:{currentprice}')
            self.__shortPos.exitMarket()

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        position.exitMarket()

    def onEnterCanceled(self, position): #cancel order before execution
        if self.__longPos is position:
            self.__longPos = None
        if self.__shortPos is position:
            self.__shortPos = None

    def onExitOk(self, position): #exit position
        execInfo = position.getExitOrder().getExecutionInfo()

        if self.__longPos is position:
            self.__longPos = None
        if self.__shortPos is position:
            self.__shortPos = None

def run_strategy(model):

    feed = csvfeed.GenericBarFeed(frequency=pyalgotrade.barfeed.Frequency.HOUR,timezone=None, maxLen=1024)
    feed.addBarsFromCSV(instrument="gbpusd",path="C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/GBPUSD60.csv") #FIXED BARFEED ISSUE
    myStrategy = machine_learning_strategy(feed, "gbpusd",1000000,model)
    myStrategy.run()
    final_portfolio_PL= myStrategy.getBroker().getEquity()-1000000
    print("Final portfolio Earning: {} in SMA Period ".format(final_portfolio_PL))
    return final_portfolio_PL

if __name__=='__main__':
    model=joblib.load('model.sav')
    record=[]
    #for period in range(1,241,5): #SMA1 to SMA240
     #   return_port=run_strategy(period)
     #   record.append((return_port,period))
    result=run_strategy(model)
    df = pd.DataFrame(record,columns=['Period', 'Earning'])
    df.to_csv('final_result.csv', index=False)
    print('FINISHED')

