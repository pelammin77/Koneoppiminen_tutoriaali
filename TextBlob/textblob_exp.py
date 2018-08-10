"""
File: textblob_exp.py
Author Petri Lammiaho
Simple TexBlob library example
Module is of my Finnish machine learning tutorial
https://www.youtube.com/playlist?list=PLH1J1mm44iNU5Zb6cXGJFZJ2_QvNBWCSK
"""

from textblob import TextBlob
text = """
The titular threat of The Blob has always struck me as the ultimate movie monster: an insatiably hungry, amoeba-like 
mass able to penetrate virtually any safeguard, capable of--as a doomed doctor chillingly describes it--"assimilating 
flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most devastating of potential consequences, not unlike 
the grey goo scenario proposed by technological theorists fearful of artificial intelligence run rampant.
"""

blob = TextBlob(text)
#print(dir(blob))
print(blob)
print(blob.words) # word tokenizer
print(blob.sentences)# sentencestokenizer
#print(len(blob.sentences))#  prints num sentences
print(blob.tags) # prints tags
print(blob.pos_tags) # prints pos tags

sentiment_pos = "This movie was great! 5 stars"
sentiment_neg = "Movie was awful. I did not like it. 1 star"
suomi_text = "Todella hyvä leffa! 5 tähteä"
#
sent_blob = TextBlob(sentiment_pos)
#
print(sent_blob.sentiment) #sentiment analysis
print(sent_blob.translate(to="fi")) # translates text english to fin

#
sent_blob = TextBlob(sentiment_neg) # neg sentence
print(sent_blob.sentiment) # neg  sentiment analysis

translate_text = TextBlob(suomi_text)
print(sent_blob.translate(to="fi"))
print(translate_text.translate(to="en"))# translates text fin to en
