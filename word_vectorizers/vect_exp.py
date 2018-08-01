"""
file: vect_exp.py
author : Petri Lamminaho
Simple text vectorizers example. Uses TFIDF and count vectorrizer
Code is part of my machine learning tutorial video  
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words

count_vect = CountVectorizer()
tfidf_vect = TfidfVectorizer()
#print(stop_words.ENGLISH_STOP_WORDS) # removes stop words

text1 = 'How are you, are you doing fine?'
text2 = "What's up?"

count_test_vect_text = count_vect.fit_transform([text1, text2]) # train count vect
tfidf_vect_text = tfidf_vect.fit_transform([text1, text2]) # train tfidf vect

## tfidf-vectorizer
print(tfidf_vect_text)
print(tfidf_vect_text.toarray())
print(tfidf_vect.get_feature_names())
print(tfidf_vect_text[0])
print("#"*20)
print(tfidf_vect_text[1])
print(tfidf_vect.inverse_transform(tfidf_vect_text[1]))
print("-"*20)

##count vectorizer
print(count_test_vect_text)
print(count_test_vect_text.toarray())
print(count_vect.get_feature_names())
print(count_test_vect_text[0])
print("#"*20)
print(count_test_vect_text[1])
print(count_vect.inverse_transform(count_test_vect_text[0]))
