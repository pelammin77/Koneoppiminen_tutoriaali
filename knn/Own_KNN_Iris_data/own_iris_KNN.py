"""
file: own_iris_KNN.py
Author:Petri Lamminaho
Simple knn algorithm with Iris data
It has simple train test split
"""
########################################################
import  pandas as pd
import random
from scipy.spatial import distance
import warnings
from collections import Counter
#########################################################
def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn('K is set to a value less than labels!')

    distances = []
    for label in data:
        for features in data[label]:
            euclidean_distance = distance.euclidean(features, predict )
            #euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, label])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
###############################################################################
iris_df = pd.read_csv('iris.data', header=None)# read data to dataframe
full_data = iris_df.values.tolist()
random.shuffle(full_data)

#Train test split
test_size = 0.2
train_set = {'Iris-setosa':[], 'Iris-versicolor':[], 'Iris-virginica':[]} #create train set
test_set ={'Iris-setosa':[], 'Iris-versicolor':[], 'Iris-virginica':[]} #create test set

train_data = full_data[:int(test_size * len(full_data))]
test_data = full_data[int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

print(train_set)

oikeat = 0
total = 0

#calculate the right answers
for label in test_set:
    for data in test_set[label]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if label == vote:
            oikeat +=1
        total +=1

print("acc:", oikeat/total) # the right answers/total
oma_horsma = [3.6, 1.9, 2.6, 2.3] #new flower
pred = test_data[10][:-1]
true_value =test_data[10][-1]
res = k_nearest_neighbors(train_set, oma_horsma) #make own predict
print("Predict:",res)



