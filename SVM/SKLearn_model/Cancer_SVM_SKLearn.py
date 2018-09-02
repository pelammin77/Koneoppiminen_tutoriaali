"""
File: Cancer_SVM_SKLearn.py
author: Petri Lamminaho
SVM-model with Scikit Learn
Model classify cancer data
(Same that KNN only algorithm changed)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from datetime import datetime
start_time = datetime.now()
df = pd.read_csv('cancer.data', header=None)
df.replace('?', -99999, inplace=True) # handle outlines
df = df.astype(float)
df.drop(df.columns[0], axis=1, inplace=True) # drop id column
X = np.array(df.iloc[:, 0:-1])# X (features) are all rows and first-last columns
y = np.array( df.iloc[:, -1]) # y (labels are all rows and last column

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50) # jaetaandata train ja test-dataan
svm_clf = svm.LinearSVC()
#svm_clf = svm.SVC(kernel="linear") # acc:0.95
svm_clf.fit(X_train, y_train)
end_time = datetime.now()
total_time = end_time-start_time

print(svm_clf.score(X_test, y_test))# test/print acc
print("Classifier takes", total_time)