"""
File: voting classifier.py
author: Petri Lamminaho
Simple voting classifier. Uses Scikit Learn's algorithm
"""

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier(random_state=1)

print('5-fold cross validation:\n')
labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Tree']
for clf, label in zip([clf1, clf2, clf3, clf4], labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ("svm", clf4) ],
                        voting='soft',
                        weights=[1, 1, 5, 1])

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Tree', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, clf4,  eclf], labels):
    scores = model_selection.cross_val_score(clf, X, y,
                                                 cv=5,
                                                 scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]"
% (scores.mean(), scores.std(), label))
eclf.fit(X,y)
pred = eclf.predict_proba(X[[0]])
print(pred)