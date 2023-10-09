import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import uniform

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def train(x_train, y_train, learning_rate, bias, weights):
    rows = len(x_train)
    for i in range(rows):
        x = x_train.values[i]
        y = y_train.values[i]
        z = np.matmul(weights.transpose(), x) + bias
        current_y = softmax(z)
        delta_z = current_y - y
        delta_weights = np.matmul(delta_z, x.transpose())
        delta_bias = delta_z
        weights = weights - learning_rate * delta_weights
        bias = bias - learning_rate * delta_bias
    return bias, weights

def train_accuracy(x_train, y_train, bias, weights):
    rows = len(x_train)
    correct = 0
    for i in range(rows):
        x = x_train.values[i]
        y = y_train.values[i]  
        z = np.matmul(weights.transpose(), x) + bias
        current_y = softmax(z)
        predicted_y = np.argmax(current_y)
        if predicted_y == y:
            correct += 1
    print("Train accuracy: ", 100.0 * correct / rows, "%")

def test(x_test, y_test, bias, weights):
    rows = len(x_test)
    correct = 0
    for i in range(rows):
        x = x_test.values[i]
        y = y_test.values[i]  
        z = np.matmul(weights.transpose(), x) + bias
        current_y = softmax(z)
        predicted_y = np.argmax(current_y)
        if predicted_y == y:
            correct += 1
    print("Test accuracy: ", 100.0 * correct / rows, "%")

def execute(wine):
    x = wine.drop('quality', axis = 1)
    y = wine['quality']
    learning_rate = 0.1
    # bias = np.random.randint(-1, 1, size=(11))
    bias = [0.1 for i in range(11)]
    weights = np.array([[uniform(-2, 2) for i in range(11)] for j in range(11)])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 16)
    for i in range(500):
        # print("Epoch: ", i)
        bias, weights = train(x_train, y_train, learning_rate, bias, weights)
    train_accuracy(x_train, y_train, bias, weights)
    test(x_test, y_test, bias, weights)
        
wine = pd.read_csv('winequality-red.csv')
execute(wine)