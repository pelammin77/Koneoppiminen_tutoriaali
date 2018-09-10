"""
File: own_SVM.py
Author Petri Lamminaho
Simple SVM algorithm from scratch
I used https://pythonprogramming.net/ SVM tutorial to make my code to work
 """

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np



style.use('ggplot')


class Own_Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'red', -1: 'blue'}
        self.marker = {-1: '_', 1:'+'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        opt_dict = {}

        transforms_matrix = [[1, 1], # all vector direct
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        help_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    help_data.append(feature)

        self.max_feature_value = max(help_data)
        self.min_feature_value = min(help_data)
        help_data = None # frees list

        # support vectors yi(xi.w+b) = 1


        step_sizes = [self.max_feature_value * 0.1, # "learning rates
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,
                      ]


        b_range_multiple = 2
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10 # first optimum value

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms_matrix:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step
                    print("Muutetaan steppiä")

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            print("Opt |w|:", self.w)
            print("Opt b",self.b)
            latest_optimum = opt_choice[0][0] + step * 2
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b) # returns dot product's sign + or -
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification]) # draw point
        return classification # returns class + or -


    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, marker=self.marker[i],  color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict_2 = {
            1: np.array([[1, 9],
                           [2, 6],
                           [3, 7], ]),

             -1: np.array([[8, 1],
                          [7, -1],
                          [9, 3], ])}



data_dict = {1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             -1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

svm = Own_Support_Vector_Machine()
svm.fit(data=data_dict)

own_predict = [
                [2, 6],
                [1, 0],
                [8, 3],

              ]
for p in own_predict:
    svm.predict(p)

svm.visualize()