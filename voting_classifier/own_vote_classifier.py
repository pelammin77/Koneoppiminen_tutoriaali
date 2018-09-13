"""
file: own_vote_classifier.py
author: Petri Lamminaho
Simple own Voting Classifier from scratch
"""

from collections import Counter
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50) # jaetaandata train ja test-dataan

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier(random_state=1)

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Tree']
votes = []

for clf, label in zip([clf1, clf2, clf3, clf4], labels):
    clf.fit(X_train, y_train)
    #scores = clf.score(X_test, y_test)

    scores = model_selection.cross_val_score(clf, X_test, y_test,
                                              cv=5,
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

for clf, label in zip([clf1, clf2, clf3, clf4], labels):
    pred = clf.predict(X_test[[0]])
    votes.append(pred[0])
vote_result = Counter(votes).most_common(1)[0][0] # result is most common feature
prodba = Counter(votes).most_common(1)[0][1] / len(labels)*100
print("Predict:",vote_result)
print("Reliable:", prodba,"%")
