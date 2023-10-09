#!/usr/bin/env python3
import math
import csv
import numpy as np
import nn

def main():
    print("Exercise")
    exercise()
    # print()
    # print("Wine classfier")
    # wine_classifier()

def exercise():
    x = np.array([1, 3, 0], dtype=np.float32)
    w = np.matrix(
        [
            [0.3,   0.1,    -2],
            [-0.6,  -0.5,   2],
            [-1,    -0.5,   0.1]
        ],
        dtype=np.float32
    )
    b = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    l = np.float32(0.1)
    y = np.array([0, 1, 0], dtype=np.float32)

    neural_network = nn.NeuralNetwork(w, b, l)

    print("Initial network")
    neural_network.display()
    print(f"Prediction for x before training: {neural_network.predict(x)}")

    neural_network.train(x, y)

    print()
    print("Edited network")
    neural_network.display()
    print(f"Prediction for x after  training: {neural_network.predict(x)}")

def wine_classifier():
    raw_data = read_csv()
    x, y = extract_data(raw_data)
    w, b = construct_weights_biases(x, y)
    l = np.float32(0.01)

    training, _validation, testing = partition(x, y)

    neural_network = nn.NeuralNetwork(w, b, l)

    correct = 0
    for values, expected_results in zip(testing[0], testing[1]):
        correct += expected_results[np.argmax(neural_network.predict(values))][0]

    print(f"Initial values:\t{correct} / {len(testing[0])}")

    epochs = 100
    for i in range(epochs):
        print(f"Training: \t{i + 1} / {epochs}", end="\r")
        for values, expected_results in zip(training[0], training[1]):
            neural_network.train(values, expected_results)
    
    print()

    correct = 0
    for values, expected_results in zip(testing[0], testing[1]):
        correct += expected_results[np.argmax(neural_network.predict(values))][0]

    print(f"Final values:\t{correct} / {len(testing[0])}")

def read_csv():
    raw_data = []

    with open("./winequality-red.csv", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            raw_data.append([np.float32(x) for x in row])

    return raw_data

def extract_data(raw_data):
    unique_features = { row[-1] for row in raw_data }

    x = [np.array(row[:-1]).reshape(-1, 1) for row in raw_data]
    y = [np.array([1 if row[-1] == value else 0 for value in unique_features]).reshape(-1, 1) for row in raw_data]

    return x, y

def construct_weights_biases(x, y):
    inputs = len(x[0])
    outputs = len(y[0])

    w = np.random.randn(inputs, outputs)
    b = np.random.randn(outputs, 1)

    return w, b

def partition(x, y):
    eightyPercent = math.floor(0.80 * len(x))
    ninetyPercent = math.floor(0.90 * len(x))

    training = (x[:eightyPercent], y[:eightyPercent])
    validation = (x[eightyPercent:ninetyPercent], y[eightyPercent:ninetyPercent])
    testing = (x[ninetyPercent:], y[ninetyPercent:])

    return training, validation, testing

if __name__ == '__main__':
    main()
