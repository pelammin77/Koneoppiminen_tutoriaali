"""
Tiedosto: tree.py
Tekijä: Petri Lamminaho
Kuvaus:  Scikit Learn-kirjastolla toteutettu  Decision tree koneoppimismalli
         -Esimerkissä käytetään Iris-datasettiä
         -Ohjelma piitää puusta  kuvan graphviz-kirjaston avulla ja tallentaa sen pdf-muotoon
"""
##############################################################################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.externals.six import StringIO
import graphviz
################################################################################
iris = load_iris() # ladataan Iris data
X = iris.data
print("Iris database length",len(X))
y =  iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50) # jaetaan data train ja test-dataan
print("Training data length",len(X_train))
clf = tree.DecisionTreeClassifier() # luodaan malli
clf.fit(X_train, y_train) # training
conf = clf.score(X_test, y_test)
print("Model conf:", conf)
################################################################################
''' Piiretään puu '''
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)
graph = graphviz.Source(dot_data.getvalue())
graph.render("iris_tree", view=True)
##################################################
# Tehdään ennustus ja testataan sitä oikeaan vastaukseen
print("Test data length",len(y_test))
print("True value:",y_test[0])
pred = clf.predict(X_test[[0]])
print("Tree predict:",pred[0])
