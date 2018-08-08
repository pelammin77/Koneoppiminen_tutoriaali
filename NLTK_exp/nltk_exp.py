"""
file: nltk_exp.py
Author: Petri Lamminaho
Simple NLTK example

"""
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from  nltk.stem import WordNetLemmatizer

#nltk.download() # downloads nltk content to computer

## Sample text
text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

# Stopwords:
stop_words = set(stopwords.words("english"))
#print(stop_words) # prints all stopwords
#
words = word_tokenize(text) #word tokenizer
sents = sent_tokenize(text) #sentence tokenizer
#
#print(words)
#print(sents)
#print(sents[0])
first_sent = word_tokenize(sents[0])
#
first_sent_str = ' '.join(first_sent)
#print(first_sent_str)
first_sent_words = word_tokenize(first_sent_str)
#print(first_sent_words)
text_sw_removed = []
#
# removes stopwords
for w in first_sent_words:
     if w not in stop_words:
         text_sw_removed.append(w)

#print(text_sw_removed)

# stemming
stem_words = ["car", "cars"]
ps = PorterStemmer()
for w in stem_words:
     print(ps.stem(w))
#
#
lem = WordNetLemmatizer() # lemmatizer
print(lem.lemmatize("cars")) # prints car

print(lem.lemmatize("better")) #Prints good
print(text_sw_removed) # prints text without stopwords 

"""
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""

pos_tags = nltk.pos_tag(text_sw_removed) # get pos tags
print(pos_tags)


"""
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
"""

from nltk import word_tokenize, pos_tag, ne_chunk
sentence = "Mark and John are working at Google."
print (ne_chunk(pos_tag(word_tokenize(sentence))))
##prints:
""" 
(PERSON Mark/NNP)
  and/CC
  (PERSON John/NNP)
  are/VBP
  working/VBG
  at/IN
  (ORGANIZATION Google/NNP)
  ./.)
"""