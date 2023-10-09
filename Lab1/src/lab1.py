

import numpy as np
from numpy import ndarray
from typing import Tuple

def compute_gradients(x: ndarray, y_pred: ndarray, y_true: ndarray) -> Tuple[ndarray, ndarray]:
    differece = y_pred - y_true
    gradient_b = differece.mean()
    matrix_mul = x.T @ differece
    gradients_w = (matrix_mul).mean(dim=1)
    return gradients_w, gradient_b

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid division by zero
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss


x = np.array([1, 3, 0]) ## input

b = np.array([0.1, 0.1, 0.1]) ## bias

y = 1


 ## weights
#w1 = np.array([0.3, 0.1, -2])
w2 = np.array([ -0.6,  -0.5, 2])
#w3 = np.array([-1, 0.5, 0.1])
          

# linear prediction
#y1 = np.dot( w1 , x) + b[0]
y2 = np.dot( w2 , x) + b[1]
#y3 = np.dot( w3 , x) + b[2]

#y_pred = np.array([y1,y2,y3])

predicted_probability = sigmoid (y2)

print( predicted_probability)
print( cross_entropy_loss (y ,  predicted_probability ) )
#print(y3)

