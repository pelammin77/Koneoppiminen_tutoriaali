"""
file: softmax.py
author: Petri Lamminaho
Simple sofmax example
"""
import numpy as np

logits = [4.0, 2.0, 0.2]
sum_of_logits = np.sum(logits)
print(sum_of_logits)
exps = [np.exp(i) for i in logits]
print(exps)
sum_of_exps = np.sum(exps)
print(sum_of_exps)
softmax = [j/sum_of_exps for j in exps]
print(softmax)
print(np.sum(softmax)) # should print 1.0


