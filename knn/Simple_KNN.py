"""
file: simple_KNN.py
Author Petri Lamminaho
Simple own KNN algorithm
"""
#############################
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.spatial import distance
import warnings
from collections import Counter
style.use('fivethirtyeight')
##################################################################
# This function makes the KNN!!
def k_nearest_neighbors(data, predict, k=3):
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
###################################################################

#test dataset
dataset = {'blue': [[1, 2 ], [2, 3], [3, 1]], 'red': [[6, 5], [7, 7] , [8, 6]]}
new_point = [3, 2]

#draw dataset points with plt
# oneliner:
#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# same with multiple lines
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1],s=100,color=i)
result = k_nearest_neighbors(dataset, new_point, 5)
print("New point is:", result)
plt.scatter(new_point[0], new_point[1], s=300, color=result)
plt.show()

