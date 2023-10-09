from sklearn.metrics import accuracy_score
from typing import Tuple

from numpy import ndarray
import numpy as np

accuracies = []


def softmax(z: ndarray) -> ndarray:
    exp_z = np.exp(z)
    y_prediction = exp_z / exp_z.sum()
    return y_prediction


def compute_prediction(x_features: ndarray, weights: ndarray, bias: ndarray) -> ndarray:
    # Linear Combinations
    z = weights.T @ x_features + bias

    # Softmax
    return softmax(z)


def main(alfa: float = 0.5):
    # Input data
    x_features = np.expand_dims(np.array([1, 3, 0]), axis=1)
    weights = np.array([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
    bias = np.expand_dims(np.array([0.1, 0.1, 0.1]), axis=1)
    y_label = np.expand_dims(np.array([0, 1, 0]), axis=1)

    # Get prediction
    y_prediction = compute_prediction(x_features, weights, bias)

    # Gradient loss with respect to z
    gradient_loss = y_prediction - y_label

    # Gradients
    delta_weights = gradient_loss @ x_features.T
    delta_bias = gradient_loss

    # print(gradient_loss.shape)
    # print(np.reshape(x_features, (1, x_features.shape[0])).shape)
    # print(x_features.T.shape)

    # Update
    weights -= alfa * delta_weights
    bias -= alfa * delta_bias

    print("Y prediction: \n", y_prediction)
    # print("Loss: ", compute_loss(y_prediction, y_label))
    print("Gradient loss: \n", gradient_loss)

    print("\nFor η: \n", alfa)

    print("\nWeights gradient: \n", delta_weights)
    print("Bias gradient: \n", delta_bias)

    print("\nFinal weights: \n", weights)
    print("Final bias: \n", bias)


main(alfa=0.5)

# What we got:
# Y prediction:
#  [[0.00405191]
#  [0.00447805]
#  [0.99147003]]
# Gradient loss:
#  [[ 0.00405191]
#  [-0.99552195]
#  [ 0.99147003]]
#
# For η:
#  0.5
#
# Weights gradient:
#  [[ 0.00405191  0.01215573  0.        ]
#  [-0.99552195 -2.98656584  0.        ]
#  [ 0.99147003  2.9744101   0.        ]]
# Bias gradient:
#  [[ 0.00405191]
#  [-0.99552195]
#  [ 0.99147003]]
#
# Final weights:
#  [[ 0.29797404  0.09392213 -2.        ]
#  [-0.10223903  0.99328292  2.        ]
#  [-1.49573502 -1.98720505  0.1       ]]
# Final bias:
#  [[ 0.09797404]
#  [ 0.59776097]
#  [-0.39573502]]
