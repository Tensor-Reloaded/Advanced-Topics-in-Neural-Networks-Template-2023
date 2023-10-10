import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    :param x: The batched input; ndarray of size [B, N], where B is the batch dimension and N is the number of features.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(y, y_p):
    return -np.sum(np.multiply(y, np.log(y_p)))


def main():
    x = np.array([1, 3, 0])
    W = np.array([[0.3, 0.1, -2],
                  [-0.6, -0.5, 2],
                  [-1, -0.5, 0.1]])

    b = np.array([0.1, 0.1, 0.1])
    y = np.array([0, 1, 0])
    learning_rate = 0.2

    z = np.dot(W.T, x) + b
    print("z = ", z)

    y_p = softmax(z)
    print("y' = ", y_p)

    loss = cross_entropy(y, y_p)
    print("loss = ", loss)

    delta_w = np.dot((y_p - y).reshape(3, 1), x.reshape((1, 3)))
    print("delta_W = \n", delta_w)

    delta_b = y_p - y

    print("delta_b = ", delta_b)

    W -= learning_rate * delta_w
    print("Adjusted W = \n", W)

    b -= learning_rate * delta_b
    print("Adjusted b = ", b)

if __name__ == '__main__':
    main()

# Program Output
# z =  [-1.4 -1.3  4.1]
# y' =  [0.00405191 0.00447805 0.99147003]
# loss =  5.408566554451391
# delta_W =
#  [[ 0.00405191  0.01215573  0.        ]
#  [-0.99552195 -2.98656584 -0.        ]
#  [ 0.99147003  2.9744101   0.        ]]
# delta_b =  [ 0.00405191 -0.99552195  0.99147003]
# Adjusted W =
#  [[ 0.29918962  0.09756885 -2.        ]
#  [-0.40089561  0.09731317  2.        ]
#  [-1.19829401 -1.09488202  0.1       ]]
# Adjusted b =  [ 0.09918962  0.29910439 -0.09829401]