import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


eur = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/EURUSD60.csv',usecols=[0,1,2,3,4,5],skiprows=1)
eur.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volumn']
eur['Date'] = eur['Date'].apply(lambda y: datetime.strptime(y, '%d.%m.%Y %H:%M:%S.000'))
eur['Date'] = eur['Date'].apply(lambda y: datetime.strftime(y, "%Y-%m-%d %H:%M:%S"))
eur['Date']=pd.to_datetime(eur['Date'])
eur.set_index('Date',inplace=True)
eur = eur.drop_duplicates(keep=False)
gbp = pd.read_csv('C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/GBPUSD60.csv',usecols=[0,1,2,3,4,5],skiprows=1)
gbp.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volumn']
gbp['Date'] = gbp['Date'].apply(lambda y: datetime.strptime(y, '%d.%m.%Y %H:%M:%S.000'))
gbp['Date'] = gbp['Date'].apply(lambda y: datetime.strftime(y, "%Y-%m-%d %H:%M:%S"))



gbp['Date']=pd.to_datetime(gbp['Date'])
gbp.set_index('Date',inplace=True)
gbp = gbp.drop_duplicates(keep=False)

gbp_close=gbp['Close'].values
eur_close=eur['Close'].values
timeframe = 24*5 #a week
start_count = 0
end_count = 0
corr_list = []
date = gbp.index.tolist()

for item in gbp_close: #
    end_count=start_count+timeframe
    try:
        gbp_period_arr = gbp_close[start_count:end_count]
        eur_period_arr = eur_close[start_count:end_count]
        corr=np.corrcoef(gbp_period_arr,eur_period_arr)
        data_date = date[end_count]
        corr_list.append((data_date,round(corr[0][1],8)))
    except:
        pass
    start_count+=1

df = pd.DataFrame(corr_list)
df.columns=['Date','Correlation']
df.set_index('Date',inplace=True)
df.to_csv(f'C:/Users/LokFung/Desktop/IERGYr4/IEFYP/EURUSDCorrelation_tf={timeframe}.csv')
"""
#graph
fig ,axis1 = plt.subplots()
color = 'tab:orange'
axis1.set_xlabel('Time')
axis1.set_ylabel('GBPUSD', color=color)
axis1.plot(gbp_close, color=color)
axis1.tick_params(axis='y', labelcolor=color)
print(gbp_close)
axis2 = axis1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
axis2.set_ylabel('EURUSD', color=color)  # we already handled the x-label with ax1
axis2.plot(eur_close, color=color)
axis2.tick_params(axis='y', labelcolor=color)
#graph

"""


#plt.title('GBPUSD and EURUSD Trend ')
#plt.legend()
#plt.show()
