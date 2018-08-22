"""
File: simple_log_regression.py
author: Petri Lamminaho
Own simple logistic regression algorithm
Generates test data
Updates gradient/ find best weights
Prints acc
and draw result
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
"""
GWnwrate test data 
"""
def generate_test_data(num_points, seed):
    np.random.seed(seed)
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_points)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_points)
    features = np.vstack((x1, x2)).astype(np.float32)
    labels = np.hstack((np.zeros(num_points),
                           np.ones(num_points)))
    return features, labels

"""
Draw data points with Mathplotlib
"""
def draw_data_points(features, labels):
    plt.figure(figsize=(12,8))
    plt.scatter(features[:, 0], features[:, 1],
                                   c = labels, alpha = .4)
    plt.show()

"""
Sigmoid function
 """
def sigmoid(z):
    return  1 / (1 + np.exp(-z))

"""
Calculating log likelihood 
"""

def log_likelhood(features, result, weigths):
    scores = np.dot(features,weigths)
    return np.sum(result * scores - np.log(1 + np.exp(scores)))

"""
regression function 
"""

def log_regression(features, true_value, num_steps, learning_rate, fit_intercept = True):
    if fit_intercept:
        intercept = np.ones((features.shape[0], 1))  #creates Matrix of ones
        features = np.hstack((intercept, features))# adds Matrix of ones  to the features

    weights = np.zeros(features.shape[1]) # creates weight 3 zero vectors [0,0,0]


    for step in range(num_steps):
        scores = np.dot(features, weights) #features and weights Dot product is score
        predictions = sigmoid(scores) # gives score to sigmoid returns value 0-1

        # Update weights with log likelihood gradient
        output_error_signal = true_value - predictions # error is difference true_value and sigmoid pred

        gradient = np.dot(features.T, output_error_signal) # Calculating gtradient
        weights += learning_rate * gradient # updating weights using gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print("training...")
            print(log_likelhood(features, true_value, weights))

    return weights
##########################################################################
random_features, random_labels = generate_test_data(10000, 10)
draw_data_points(random_features, random_labels)

weights = log_regression(random_features, random_labels,
                     num_steps = 50000, learning_rate = 5e-5, fit_intercept=True)
print ("OWN LOGISTIC REGRESSION WEIGHTS => ")
print("w0:", weights[0], "w1:", weights[1], "w2:", weights[2])

final_scores = np.dot(np.hstack((np.ones((random_features.shape[0], 1)),
                                 random_features)), weights)
preds = np.round(sigmoid(final_scores))

clf = LogisticRegression(fit_intercept=True, C = 1e15) # create sklearn logistic regression
clf.fit(random_features, random_labels)

print ('Accuracy own algorithm : {0}'.format((preds == random_labels).sum().astype(float) / len(preds)))
print("Sklearn weights are:", clf.intercept_, clf.coef_)# prints sklearn weights

print("Sckit Learn Acc:", clf.score(random_features, random_labels))

plt.figure()
plt.scatter(random_features[:, 0], random_features[:, 1],
             c = preds == random_labels - 1, alpha = .8, s = 50)
plt.show()
