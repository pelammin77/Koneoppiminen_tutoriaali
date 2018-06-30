"""
Tiedosto: forest.py
Tekijä: Petri Lamminaho
Kuvaus: Scikit Learn-kirjastolla luotu Random Forest esimerkki
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
###########################################################
def lataa_data_pandaan(file_name):
    # lataa datan Pandas dataframeen ja palauttaa sen
   # import pandas as pd
    df = pd.read_csv(file_name, sep=',')
    return df
##########################################################################
data = lataa_data_pandaan("data.csv")

X = data.iloc[:, 0: -1]
y = data.iloc[:, -1]
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1080)

#luodaan kolme metsää
forest1 = RandomForestClassifier(n_estimators=1)
forest2 = RandomForestClassifier(n_estimators=5)
forest3 = RandomForestClassifier(n_estimators=10)

#treeni
forest1.fit(X_train, y_train)
forest2.fit(X_train, y_train)
forest3. fit(X_train, y_train)

#lasketaan mallin tarkuus
forest1_acc = forest1.score(X_test, y_test)
forest2_acc = forest2.score(X_test, y_test)
forest3_acc = forest3.score(X_test, y_test)

#tulostetaan mallin tarkuus
print("Forest1 acc(1 tree):",forest1_acc)
print("Forest2 acc(5 trees):",forest2_acc)
print("Forest3 acc(10 trees:)",forest3_acc)
