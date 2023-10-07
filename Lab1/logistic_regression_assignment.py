from numpy import exp
import numpy as np

def softmax(vector):
    e = exp(vector)
    return e / e.sum()

def logisticRegression(x,W,b,y,n):
    z=np.dot(W,x)+b
    #predicted probability 
    y_pred=softmax(z)
    
    #the loss vector
    loss = -(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    #total loss
    total_loss = np.sum(loss)

    #gradients
    delta_z = y_pred - y
    delta_W = np.outer(delta_z, x)
    delta_b = delta_z

    #updated values for weights matrix and biases based on the learning rate n
    W -= n * delta_W
    b -= n * delta_b

    print("Predicted probability: ", y_pred)
    print("Losses: ", loss)
    print("Total loss: ", total_loss)
    print("Weights:")
    print(W)
    print("Biases:", b)

x = np.array([1, 3, 0])
W = np.array([[0.3, 0.1, -2],
              [-0.6, -0.5, 2],
              [-1, -0.5, 0.1]])
b = np.array([0.1, 0.1, 0.1])
y_true = np.array([0, 1, 0])

#learning rate = ?
n = 0.1

logisticRegression(x, W, b, y_true, n)
