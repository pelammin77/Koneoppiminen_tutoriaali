"""
file: KNN_exp.py
Author: Petri Lamminaho
Simple knn model. Uses Iris dataset
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
##################################################
iris = load_iris() # loading data
X = iris.data
y = iris.target
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.20, random_state=10) # train test split
knn_model = KNeighborsClassifier(n_neighbors=3) # create the model
knn_model.fit(X_training, y_training)# training
acc = knn_model.score(X_testing, y_testing)# calculate model's acc
print("Model acc:", acc) # printing acc
#######################################################################

#testing
print("True value:",y_testing[0])
pred = knn_model.predict(X_testing[[0]])
print("KNN predict:",pred[0])
