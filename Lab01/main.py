import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def cross_entropy_loss(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

class Classifier:

    def __init__(self):
        self.weights = np.random.randn(6, 11) / np.sqrt(11)
        self.biases = np.random.randn(1, 6)
        self.lr = 0.05

    def train(self):
        for epoch in range(0, 30):
            for _ in range(0, len(train_x)):
                # print(_)
                x = np.array([train_x[_]])
                z = np.dot(x, np.transpose(self.weights)) + self.biases
                y = np.array([train_y[_]])
                y_hat = softmax(z)
                loss = cross_entropy_loss(y_hat, train_y[_])
                gradient_ = np.array(y_hat - y)
                weight_gradient = np.transpose(gradient_) * np.ones((6, 11)) * train_x[_]
                bias_gradient = gradient_
                self.weights = self.weights - self.lr * weight_gradient
                self.biases = self.biases - self.lr * bias_gradient

    def predict(self):
        correct = 0
        for _ in range(0, len(test_x)):
            x = np.array([test_x[_]])
            z = np.dot(x, np.transpose(self.weights)) + self.biases
            y = np.array([train_y[_]])
            y_hat = softmax(z)
            correct += (np.argmax(y) == np.argmax(y_hat))
        print("Correct classified test data  {}%.".format(correct * 100 / len(test_x)))

    def custom(self, x, w, b, y, lr, it):
        for _ in range(0, it):
            z = np.dot(w, np.transpose(x)) + b
            y_hat = softmax(z)
            loss = cross_entropy_loss(y_hat, y)
            gradient_ = np.array(y_hat - y)
            weight_gradient = np.ones((3, 3)) * gradient_ * x
            bias_gradient = gradient_
            w = w - lr * weight_gradient
            b = b - lr * bias_gradient
            if _ == 0:
                print("Initial loss is {}".format(loss))
            elif _ + 1 == it:
                print("Final loss is {}".format(loss))


def prepare_data_set():
    # returns train and test data normalized in [0, 1] as numpy arrays
    df = pd.read_csv('winequality-red.csv')
    x = df.drop(['quality'], axis=1)
    y = df['quality']

    data = asarray([[x] for x in y])
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.1, random_state=27)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = prepare_data_set()
    classifier = Classifier()
    weights = np.array([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
    inp = np.array([1, 3, 0])
    bias = np.array([0.1, 0.1, 0.1])
    out = np.array([0, 1, 0])
    learning_rate = 0.05
    steps = 10000
    classifier.custom(inp, weights, bias, out, learning_rate, steps)
    classifier.train()
    classifier.predict()