import torch
import torchvision


def create_input_data(m_inp, features_size, output_labels_size):
    x_inp = torch.rand(m_inp, features_size)
    w_rand = torch.rand(features_size, output_labels_size)
    b_rand = torch.rand(output_labels_size)
    y_true_rand = torch.randint(10, (m_inp, output_labels_size))
    return x_inp, w_rand, b_rand, y_true_rand


def sigmoid_func(matrix):
    n = matrix.shape[0]
    m = matrix.shape[1]
    for i in range(n):
        for j in range(m):
            matrix[i][j] = 1 / (1 + torch.exp(-matrix[i][j]))
    return matrix


def gradient_decent(W, X, b, mu, error):
    W_updated = torch.add(W, torch.matmul(torch.transpose(torch.mul(mu, X), 0, 1), error))
    b_updated = torch.add(b, torch.mul(mu, error.mean(axis=0)))
    return W_updated, b_updated


def train_perceptron(X, W, b, y_true, mu):
    # Return the updated W and b
    nr_epochs = 20

    for epoch in range(nr_epochs):
        z = torch.add(torch.matmul(X, W), b)
        y = sigmoid_func(z)

        error = torch.subtract(y_true, y)

        W, b = gradient_decent(W, X, b, mu, error)

    return W, b


def test_perceptron(X, W, b, y_true):
    z = torch.add(torch.matmul(X, W), b)
    y = sigmoid_func(z)
    total_test = y_true.size(dim=0)
    correct_prediction = 0
    predicted_label = torch.argmax(y, dim=1)
    for i in range(total_test):
        if y_true[i] == predicted_label[i]:
            correct_prediction += 1
    return correct_prediction / total_test


if __name__ == '__main__':
    m = 6000
    input_size = 784
    output_size = 10
    mu = 0.05

    # The randomly created data for the homework

    # X, W, b, y_true = create_input_data(m, input_size, output_size)
    # W_updated, b_updated = train_perceptron(X, W, b, y_true, mu)

    # Loading the MNIST Dataset for the Extra part of the homework
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=6000, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=True)

    W_updated = torch.rand(input_size, output_size)
    b_updated = torch.rand(output_size)

    X_test = None
    Y_test = None

    for test_data, test_target in test_loader:
        X_test = test_data.reshape(1000, -1)
        Y_test = test_target
        break

    # Test the accuracy before training the network
    accuracy = test_perceptron(X_test, W_updated, b_updated, Y_test)
    print("Accuracy before training: {0}".format(accuracy))

    # Get the training data
    for train_data, train_target in train_loader:
        # I reshape the received data to conform to those in the neural network
        X = train_data.reshape(6000, -1)
        y_true = torch.zeros(6000, 10)
        for i, j in enumerate(train_target):
            y_true[i][j] = 1

        # Randomly initialized weights and bias
        W = torch.rand(input_size, output_size)
        b = torch.rand(output_size)

        # Get the updated weights and bias after a certain number of epochs
        W_updated, b_updated = train_perceptron(X, W, b, y_true, mu)
        break

    # Test the accuracy after training the network
    accuracy = test_perceptron(X_test, W_updated, b_updated, Y_test)
    print("Accuracy after training: {0}".format(accuracy))
