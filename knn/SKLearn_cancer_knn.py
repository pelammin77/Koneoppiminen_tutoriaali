"""
SKLearn_cancer_knn.py
Author: Petri Lamminaho 
Simple: Scikit Learn model. Model uses Breast Cancer Wisconsin (Diagnostic) Data Set
You can find the dataset from: 
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv('cancer.data', header=None) 
df.replace('?', -99999, inplace=True) # handle outlines
df = df.astype(float)
df.drop(df.columns[0], axis=1, inplace=True) # drop id column 
X = np.array(df.iloc[:, 0:-1])# X (features) are all rows and first-last columns
y = np.array( df.iloc[:, -1]) # y (labels are all rows and last column

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50) # jaetaandata train ja test-dataan
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))# test/print acc
koe = np.array([[1,2,1,1,1,2,3,2,1],[8,10,8,1,1,7,3,2,1], [2,2,3,3,1,2,3,2,1] ])# own samples 
#koe = koe.reshape(len(koe),-1)
print(knn.predict(koe)) # makes predict with own sample

