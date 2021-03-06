"""
file: spam_classifier.py
Author: Petri Lamminaho
Simple spam classifier uses Naive bayes algorithm
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

message_1 = ["Hi how are you? Can we meet tomorrow? Love Pete"] # ham
message_2 = ["Congrats! You won this great price."] #spam
all_messages = ["Hi how are you? Can we meet tomorrow? Love Pete", "Congrats! You won this great price."]
df = pd.read_csv('smsspam.csv', sep='\t', names=['class', 'text' ]) # load dataframe from csv-file
vect = CountVectorizer() # count vectors
tfIdf_vect = TfidfVectorizer(stop_words="english", min_df=10)

X = vect.fit_transform(df['text'].tolist())
y = df['class'].tolist()#y_test[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=50)
clf = MultinomialNB() # create NB-model
clf.fit(X_train, y_train) # fit own text to vectors
acc = clf.score(X_test, y_test) # calculate model acc

print("Acc:", acc)

test_index = 100
pred = clf.predict(X_test[test_index])
print("True class:", y_test[test_index])
print("Model's predict:", pred)

messages_vect = vect.transform(all_messages)
print(clf.predict(messages_vect))
