import numpy as np

W =np.array([[0.3,0.1,-2],[-0.6,-0.5,2],[-1,-0.5,0.1]])

x=np.array([1,3,0])
b=np.array([0.1,0.1,0.1])
y=np.array([0,1,0])
learning_rate = 0.01

def softmax(n):
    exp_n = np.exp(n - np.max(n))
    return exp_n / exp_n.sum()


z = np.dot(W.T, x) + b

print("Linear combinations z ")
print(z)

y_pred = softmax(z)
print("\nPredicted probabilities ")
print(y_pred)

gradient_loss_z = y_pred - y
print("\nGradient of the loss with respect to z")
print(gradient_loss_z)

gradient_loss_w = np.dot(gradient_loss_z,x.T)
print("\nGradients with respect to the weights W ")
print(gradient_loss_w)

gradient_loss_b = gradient_loss_z
print("\nGradients with respect to the biasses b ")
print(gradient_loss_b)

W -= learning_rate * gradient_loss_w
b -= learning_rate * gradient_loss_b

print("\nUpdated Weight Matrix W:")
print(W)
print("\nUpdated Bias Vector B:")
print(b)