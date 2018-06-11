"""
Tiedosto finance_ana.py
tekijä: Petri Lamminaho
Kuvaus: Yksinkertainen pörssikurssien analysointiin/ennustamiseen
        käytettävä scripti
        Lukee dataa Yahoo finance palvelusta ja yrittää arvioida osakkeen hinnan annettuna päivänä
        Koodi osa Youtubessa pitämääni koneoppimis tutoriaalia
        https://www.youtube.com/playlist?list=PLH1J1mm44iNU5Zb6cXGJFZJ2_QvNBWCSK
"""

import numpy as np
import pandas_datareader as pdr
import datetime as dt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

style.use('ggplot')


start_day = dt.datetime(2018,1,1)# osakkeen seuranta alkaaa
end_day = dt.datetime.now() # seuranta loppuu nykyhetki
df = pdr.DataReader('NOK', 'yahoo', start_day, end_day)# ladataan osakekurssit Yahoo financesta

df =df[["Adj Close"]] #
print(df.tail())


df.fillna(value=-99999, inplace=True)# asetetaan nan arvot outline pisteiksi
forecast_out = 5 # monenko päivän päähän tehdään ennustus

df['Prediction'] = df[['Adj Close']].shift(-forecast_out)# luodaan prediction sarake

X = np.array(df.drop(['Prediction'],1))#X:n otetaan kaikki muut sarakkeet paitsi Prediction
X = preprocessing.scale(X) #Scalataan X

X_forecast = X[-forecast_out:]
X =X[:-forecast_out]

y = np.array(df['Prediction']) # y on pelkästään Prediction sarake
y = y[:-forecast_out] # Datan alusta tulevaisuuteen

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model= LinearRegression()# luodaan malli
model.fit(X_train, y_train)# treeniä

conf = model.score(X_test, y_test)# lasketaan mallin luotettavuus
print()
print()
print()
print("Model's confidence is", conf)
pred = model.predict(X_forecast) # ennustetaan

###### Piirretään ennuste viiva selitys katso video  ###################
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
df['Forecast'] = np.nan
for i in pred:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

pred_df = df[['Forecast']]
print()
print()
print()
print("Forecasting stock prices", forecast_out, "days")
print("==========================================================")
print(pred_df.tail(forecast_out))

plt.plot(df['Adj Close'])
plt.plot(df['Forecast'])
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()