
from torch import Tensor
import numpy as np
from numpy import ndarray
from typing import Tuple

def compute_gradients(x: ndarray, y_pred: ndarray, y_true: ndarray) -> Tuple[ndarray, ndarray]:
    """
    :param x: The batched input; ndarray of size [B, N], where B is the batch dimension and N is the number of features.
    :param y_pred: The batched predicted labels; ndarray of size [B].
    :param y_true: The batched true labels; ndarray of size [B].
    """
    differece = y_pred - y_true
    gradient_b = differece.mean()
    gradients_w = (x.T @ differece) / len(y_pred)
    return gradients_w, gradient_b

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_probs = exp_x / sum_exp_x
    return softmax_probs

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid division by zero
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss

x = np.array([1, 3, 0]) ## input
b = np.array([0.1, 0.1, 0.1]) ## bias
y = np.array([0, 1 , 0])

 ## weights
w1 = np.array([0.3, 0.1, -2])
w2 = np.array([ -0.6,  -0.5, 2])
w3 = np.array([-1, -0.5, 0.1])

i = 0
while ( i < 50):
    i += 1
    # linear prediction
    y1 = np.dot( w1 , x) + b[0]
    y2 = np.dot( w2 , x) + b[1]
    y3 = np.dot( w3 , x) + b[2]

    prob_array = np.array([y1, y2, y3])

    y_pred = softmax(prob_array) #we get the probability distribution for belnging to any one class

    #calculate the cross entropy

    loss1 = cross_entropy_loss (y[0] ,  y_pred[0] )
    loss2 = cross_entropy_loss (y[1] ,  y_pred[1] )
    loss3 = cross_entropy_loss (y[2] ,  y_pred[2] )

    print( y_pred)

    [gradients_w, gradient_b] = compute_gradients(x, y_pred, y)

    #print(gradients_w )
    # print(gradient_b )

    learning_rate = 0.5
    # Update weights and bias
    w1 -= learning_rate * gradients_w
    w2 -= learning_rate * gradients_w
    w3 -= learning_rate * gradients_w

    b -= learning_rate * gradient_b