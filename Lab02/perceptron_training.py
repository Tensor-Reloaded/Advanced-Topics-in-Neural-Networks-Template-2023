import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def train_perceptron(X, W, b, y_true, learning_rate):
    # Forward Propagation
    z = X @ W + b
    y_pred = sigmoid(z)
    print("Predicted Output:")
    print(y_pred)

    error = y_true - y_pred
    print("Error:")
    print(error)

    # Backward Propagation
    dW = X.T @ error
    db = error.sum(dim=0)

    # Update weights and biases
    W += learning_rate * dW
    b += learning_rate * db

    return W, b

m = 100
input_size = 784
output_size = 10

W = torch.randn((input_size, output_size)) * 0.01
b = torch.zeros(output_size)

y_true = torch.zeros((m, output_size))
y_true[torch.arange(m), torch.randint(0, output_size, (m,))] = 1

learning_rate = 0.01
X = torch.rand((m, input_size))

W, b = train_perceptron(X, W, b, y_true, learning_rate)