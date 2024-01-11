# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 21:23:44 2022

@author: shash
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 13:55:03 2022

@author: shash
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 23:28:32 2022

@author: shash
"""

from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np



os.chdir("D:/Dipti/capstone")
FullRaw = pd.read_csv("Data_29-06-22.csv")

##############
# Check for NA values
###############

FullRaw.isnull().sum()

FullRaw2 = FullRaw.groupby("Date")['VALUE'].sum().reset_index().copy()


FullRaw2['Date'] = pd.to_datetime(FullRaw2['Date'])
FullRaw2.sort_values("Date", inplace = True)
FullRaw2.set_index('Date', inplace=True) 

FullRaw2.head(5)

FullRaw2.tail(5)

FullRaw2.index.min()
FullRaw2.index.max()



import seaborn as sns
sns.lineplot(data = FullRaw2)


#whitenoise

from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)
# summary stats
print(series.describe())
# line plot
series.plot()
pyplot.show()




#ramdom walk


from random import seed
from random import random
from matplotlib import pyplot
seed(1)
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
	random_walk.append(value)
pyplot.plot(random_walk)
pyplot.show()




from statsmodels.tsa.seasonal import seasonal_decompose
Decomposed_Series = seasonal_decompose(FullRaw2)
Decomposed_Series.plot()

# addiditive/Multiplicative decomposition

# additive


Decomposed_Series = seasonal_decompose(FullRaw2, model='additive', freq=10)
Decomposed_Series.plot()

# Multicative

Decomposed_Series = seasonal_decompose(FullRaw2, model='multiplicative', freq=10)
Decomposed_Series.plot()

# AD fuller test


from statsmodels.tsa.stattools import adfuller
print ("Results of Dickey-Fuller Test:")
dftest = adfuller(FullRaw2["VALUE"], autolag = "AIC")
    
dfoutput = pd.Series(dftest[0:4], index = ["Test Statistic", "p-value", "#Lags Used","No of observations Used"])
for Date, value in dftest[4].items():
    dfoutput["Critical Value (%s)" %Date] = value
    
print (dfoutput)




from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf (FullRaw2, nlags=22)
lag_pacf = pacf (FullRaw2, nlags=22)

import statsmodels.api as sm
fig = plt.figure(figsize = (12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(FullRaw2.dropna(),lags=22, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(FullRaw2.dropna(),lags=22, ax=ax2)



Train = FullRaw2[:36].copy() # First 3 years for training (2014-2016)
Test = FullRaw2[36:].copy() # Last 1 year for testing (2017)
Test

SES = SimpleExpSmoothing(Train).fit(smoothing_level=0.01) # Model building
SES.summary()
Forecast = SES.forecast(12).rename('Forecast') # Model Forecasting
Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1) 

## Plot
sns.lineplot(data = Actual_Forecast_Df)


## Validation
import numpy as np
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['VALUE'] - Validation_Df['Forecast'])/Validation_Df['VALUE'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['VALUE'] - Validation_Df['Forecast'])**2)) # RMSE



DES = Holt(Train).fit(smoothing_level=0.01, smoothing_slope=0.6)
DES.summary()
Forecast = DES.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)


## Validation
Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['VALUE'] - Validation_Df['Forecast'])/Validation_Df['VALUE'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['VALUE'] - Validation_Df['Forecast'])**2)) # RMSE


TES = ExponentialSmoothing(Train, 
                           seasonal_periods=12, 
                           seasonal='add',
                           trend = 'add').fit(smoothing_level=0.01, 
                                      smoothing_slope=0.1, 
                                      smoothing_seasonal = 0.3) 
TES.summary()
# trend = 'add' means additive trend. Use this when trend is NOT exponentially increasing/decreasing, 
# like a steep increase or decrease 
# seasonal = 'add' means additive seasonality. Use this when seasonality is NOT increasing/decreasing in magnitude
Forecast = TES.forecast(12).rename('Forecast')
Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1)

## Plot
sns.lineplot(data = Actual_Forecast_Df)


Validation_Df = Actual_Forecast_Df[-12:].copy()
np.mean(abs(Validation_Df['VALUE'] - Validation_Df['Forecast'])/Validation_Df['VALUE'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['VALUE'] - Validation_Df['Forecast'])**2)) # RMSE




# ARIMA Manual Model
###############

pip install pmdarima
from pmdarima.arima import ARIMA



## Model
arimaModel = ARIMA((6,0,12), (1,0,0,12)).fit(Train)
Forecast = pd.Series(arimaModel.predict(10)).rename('Forecast')
Forecast
Forecast.index = Test.index # Needed for the pd.concat to work correctly in the next line
Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1)
Actual_Forecast_Df

## Plot
sns.lineplot(data = Actual_Forecast_Df)

## Validation
Validation_Df = Actual_Forecast_Df[-10:].copy()
np.mean(abs(Validation_Df['VALUE'] - Validation_Df['Forecast'])/Validation_Df['VALUE'])*100 # MAPE
np.sqrt(np.mean((Validation_Df['VALUE'] - Validation_Df['Forecast'])**2)) # RMSE



# Set the correct dates as index of the forecast obtained in the previous line
start = "2018-04-01" # Check the order/ date format in FullRaw3.index. Its Year-Month-Day.
end = "2019-01-01"
futureDateRange = pd.date_range(start, end, freq='MS')
futureDateRange
Forecast.index =  futureDateRange 

Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1) # Column wise binding

Actual_Forecast_Df

## Plot
sns.lineplot(data = Actual_Forecast_Df)



###############
# SARIMA Model
###############



p = range(2)
d = range(2)
q = range(2)
P = range(2)
D = range(2)
Q = range(2)

pList = []
dList = []
qList = []
PList = []
DList = []
QList = []
mapeList = []


for i in p:
    for j in d:
        for k in q:
            for I in P:
                for J in D:
                    for K in Q:
            
                        print(i,j,k, I, J, K)
                        tempArimaModel = ARIMA((i,j,k), (I,J,K,12)).fit(Train)
                        
                        Forecast = pd.Series(tempArimaModel.predict(10)).rename('Forecast')
                        Forecast.index = Test.index
                        Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1)
                        Validation_Df = Actual_Forecast_Df[-12:].copy()
                        tempMAPE = np.mean(abs(Validation_Df['VALUE'] - Validation_Df['Forecast'])/Validation_Df['VALUE'])*100 # MAPE
                        
                        pList.append(i)
                        dList.append(j)
                        qList.append(k)
                        PList.append(I)
                        DList.append(J)
                        QList.append(K)
                        mapeList.append(tempMAPE)
            
            
arimaEvaluationDf = pd.DataFrame({"p": pList,
                             "d": dList,
                             "q": qList,
                             "P": PList,
                             "D": DList,
                             "Q": QList,
                             "MAPE": mapeList})        
            


###############






arimFinalModel = ARIMA((1,0,0), (1,1,1,12)).fit(FullRaw2) # Fullraw3 is 2014 - 2017
arimFinalModel


## Forecasting
Forecast = pd.Series(arimFinalModel.predict(12)).rename('Forecast') # Year 2018

# Set the correct dates as index of the forecast obtained in the previous line
start = "2018-04-01" # Check the order/ date format in FullRaw3.index. Its Year-Month-Day.
end = "2019-03-01"
futureDateRange = pd.date_range(start, end, freq='MS')
futureDateRange
Forecast.index =  futureDateRange 

Actual_Forecast_Df = pd.concat([FullRaw2, Forecast], axis = 1) # Column wise binding
Actual_Forecast_Df
## Plot
sns.lineplot(data = Actual_Forecast_Df)















