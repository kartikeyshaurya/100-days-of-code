import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
​
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)
train = pd.read_csv('data.csv')
values = train.values
y = values[...,2]
X = values[...,:2]
​
def stepFunction(t):
    if t >= 0:
        return 1
    return 0
​
def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])
​
# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
​
​
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
        #lines.append((W, b))
        # be w1x1 + w2x2 + b = 0 the line
        # w1x1 + w2x2 = -b
        # (w1 / w2) x1 + x2 = (-b / w2)
        # x2 = (-b / w2) - (w1 / w2) x1
        # default line y = ax + b
        # w1x1 + b = -w2x2
        # (w1x1 + b) / -w2 = x2
    return boundary_lines
​
​
boundary_lines = trainPerceptronAlgorithm(X, y)
for i in range(len(X)):
    if y[i] == 1:
        plt.plot(X[i][0], X[i][1], 'bo')
    else:
        plt.plot(X[i][0], X[i][1], 'ro')
plt.ylabel('some numbers')
​
x_min, x_max = min(X.T[0]), max(X.T[0])
​
​
def calculateX2(param, line):
    # x2 = (-b / w2) - (w1 / w2) x1
    return line[1] + line[0] * param
​
​
for i in range(len(boundary_lines)):
    plt.plot([0, 1], [calculateX2(0, boundary_lines[i]), calculateX2(1, boundary_lines[i])])
​
plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.show()