"""
file: own_iris_KNN.py
Author: Petri Lamminaho
Simple knn algorithm with bigger data
You can find the dataset from:
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
"""
import  pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import warnings
from collections import Counter

## own KNN algorithm
def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than labels!')
    distances = []
    for label in data:
        for features in data[label]:
            euclidean_distance = distance.euclidean(features, predict) # calculating points distance use Scipy
            #euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) # uses Numpy both works
            distances.append([euclidean_distance, label])
    votes = [i[1] for i in sorted(distances)[:k]] # find k:s nearest points
    vote_result = Counter(votes).most_common(1)[0][0] # result is most common feature
    return vote_result


## load and preprocessing data
df = pd.read_csv('cancer.data', header=None)
df.replace('?', -99999, inplace=True) #handles outline points
df.drop(df.columns[0], axis=1, inplace=True) # drop id column
full_data = df.astype(float).values.tolist() # all data to float
#print(full_data)
random.shuffle(full_data) # shuffle the data
#print(full_data)

##creating train/test set/data
test_size = 0.2
train_set = {2:[], 4:[]}
test_set ={2:[], 4:[]}
train_data = full_data[:int(test_size * len(full_data))]
test_data = full_data[int(test_size * len(full_data)):]
# add data to sets
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])
#print(train_set)

## testing
oikeat = 0 # num of correct prediction
total = 0

#calculating model's acc
for label in test_set:
    for data in test_set[label]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if label == vote:
            oikeat +=1
        total +=1

#make own pred
print("acc:", oikeat/total) # prints acc
koe = [2,2,4,3,3,4,2,2,1] # gives own sample to model
pred = k_nearest_neighbors(train_set,koe, k=5) # makes prediction
print(pred) # prints result
