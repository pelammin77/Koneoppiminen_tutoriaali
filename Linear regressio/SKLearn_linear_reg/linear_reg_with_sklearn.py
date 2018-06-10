"""
tiedosto: linear_reg_with_sklearn.py
Tekijä: Petri Lamminaho
Kuvaus: Lineaari regressio toteutettuna Scikit Learn-kirjastolla
        -Luetaan ensin data csv-tiedostosta Pandas-kirjaston avulla
        -Jaetaan data training/testing dataan
        -luodaan malli
        -treenataan malli
        -tehdään ennustus
        - Piirretään kuva
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#read data
dataframe = pd.read_csv('data.csv', sep=',' ) #luodaan pandas dataframe csv-tiedostosta
print(dataframe.columns)

print(dataframe.head())#tulostaa 5 ensimmäistä rtiviä datasta
x_values = dataframe[['x']]
y_values = dataframe[['y']]

"""
Jaetaan data training ja testausdataan testi 20%
"""
X_train, X_test, Y_train, Y_test = train_test_split(x_values, y_values, test_size=0.20, random_state=50)
model = linear_model.LinearRegression()# luodaan malli
model.fit(X_train, Y_train) # treeniä
#
# #visualize results
print(model.score(X_test,Y_test))# tulostetaan mallin luotettavuus/tarkkuus
plt.scatter(x_values, y_values)# piirretään datapisteet
plt.plot(x_values, model.predict(x_values))# piirretään graafi
print(model.predict(50))#tehdään ennustus
plt.scatter(50, model.predict(50)) # piirretään ennustuspiste
plt.show() # näytetään piirretyt

