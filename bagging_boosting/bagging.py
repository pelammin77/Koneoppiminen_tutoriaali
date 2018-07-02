# -*- coding: cp1252 -*-
"""
tiedosto: bagging.py
Trkijä Petri Lamminaho
Bagging ja boosting algoritmien testailua 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.datasets import load_iris

data = pd.read_csv('train.csv').as_matrix()
X = data[:,1:]
y = data[:,0]
#print(data.iloc[0])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50) # jaetaandata train ja test-dataan

puu = DecisionTreeClassifier()
puu.fit(X_train, y_train)
puu_acc = puu.score(X_test, y_test)
print("Tree acc:", puu_acc)


# Bagging algoritmi puulle 
bag = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
bag.fit(X_train, y_train)
bag_acc = bag.score(X_test, y_test)
print("Bagging acc:", bag_acc)

# boosting algoritmi puulle 
adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 20, learning_rate = 1)
adb.fit(X_train, y_train)
boosting_acc = adb.score(X_train, y_train)
print("Boosting acc",boosting_acc)





