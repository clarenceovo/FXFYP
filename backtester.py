from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.barfeed import csvfeed , yahoofeed
from pyalgotrade.technical import cross
from sklearn.externals import joblib
import talib
import pandas as pd
import pyalgotrade
demo_cash =100000


class machine_learning_strategy(strategy.BacktestingStrategy):

    def __init__(self, feed,instrument ,SMA,initialCash): #initializing order
        super(machine_learning_strategy, self).__init__(feed,initialCash)
        # We want a 15 period SMA over the closing prices.
        self.__instrument=instrument
        self.__longPos = None
        self.__shortPos = None
        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(),SMA)


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


    def onBars(self, bars): #amend and implement the strategy in this function
        bar = bars[self.__instrument]
        #self.info(bar.getClose())
        if self.__sma[-1] is None: #Wait to get enough info for SMA
            return

        if self.__longPos is None: #if no long position
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice())*100000)) #lot size is determined by the amount of cash we have in the demo
            if bar.getPrice() > round(self.__sma[-1],7): #round the 7 digits
                #elf.info('LONG')
                self.__longPos = self.enterLong(self.__instrument,lot_size*10000,True)

        elif bar.getPrice() < self.__sma[-1] and not self.__longPos.exitActive():  # EXIT RULE
            self.__longPos.exitMarket()


        if self.__shortPos is None:
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice()) * 100000))
            if bar.getPrice() < round(self.__sma[-1],7):
                self.__shortPos = self.enterShort(self.__instrument, lot_size*10000, True)
                #self.info('SHORT')

        elif bar.getPrice() > self.__sma[-1] and not self.__shortPos.exitActive():  # EXIT RULE
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

def run_strategy(period):

    feed = csvfeed.GenericBarFeed(frequency=pyalgotrade.barfeed.Frequency.HOUR,timezone=None, maxLen=1024)
    feed.addBarsFromCSV(instrument="gbpusd",path="C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/GBPUSD60.csv") #FIXED BARFEED ISSUE
    myStrategy = machine_learning_strategy(feed, "gbpusd",period,1000000)
    myStrategy.run()
    final_portfolio_PL= myStrategy.getBroker().getEquity()-1000000
    print("Final portfolio Earning: {} in SMA Period {}".format(final_portfolio_PL,period))
    return final_portfolio_PL

if __name__=='__main__':
    #model=joblib.load('model.sav')
    record=[]
    for period in range(1,241): #SMA1 to SMA240
        return_port=run_strategy(period)
        record.append((return_port,period))
    df = pd.DataFrame(record,columns=['Period', 'Earning'])
    df.to_csv('final_result.csv', index=False)
    print('FINISHED')

