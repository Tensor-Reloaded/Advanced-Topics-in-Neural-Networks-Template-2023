from pathlib import Path
from torchvision import datasets
import torch
import numpy as np

class Network:
    def __init__(self, sizes=[784, 100, 10], epochs=10, lr=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        input_layer = sizes[0]
        hidden = sizes[1]
        output_layer = sizes[2]

        self.params = {
            'W1': torch.tensor(np.random.uniform(-0.5, 0.5, (hidden, input_layer)), dtype=float),  # 100x784
            'W2': torch.tensor(np.random.uniform(-0.5, 0.5, (output_layer, hidden)), dtype=float)  # 10x100
        }

    def sigmoid_function(self, x, derivative=False):
        if derivative:
            return (torch.exp(-x)) / ((torch.exp(-x) + 1) ** 2)
        return 1 / (1 + torch.exp(-x))

    def softmax(self, x, derivative=False):
        exps = torch.exp(x - x.max())
        if derivative:
            return exps / torch.sum(exps, axis=0) * (1 - exps / torch.sum(exps, axis=0))
        return exps / torch.sum(exps, axis=0)

    def forward_propagation(self, x_train, device):
        params = self.params

        params['A0'] = x_train  # dim = 784x1
        params['A0'].to(device)

        # input layer to hidden_1

        params['Z1'] = (torch.matmul((params['W1']).to(device), params['A0'])).to(device)

        params['A1'] = (self.sigmoid_function(params['Z1'])).to(device)

        # between hidden layer and output layer
        params['Z2'] = (torch.matmul(params['W2'].to(device), params['A1'])).to(device)

        params['A2'] = (self.softmax(params['Z2'])).to(device)

        return params['A2']

    def backward_pass(self, y_train, output, device):
        params = self.params
        change_w = {}

        output = output.to(device)
        y_train = y_train.to(device)

        # Compute W2 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z2'], derivative=True)
        change_w['W2'] = (torch.outer(error, params['A1'])).to(device)

        # Compute W1 update
        transposed_w2 = (torch.transpose(params['W2'], 0, 1)).to(device)
        error = torch.matmul(transposed_w2, error) * self.sigmoid_function(params['Z1'], derivative=True)
        change_w['W1'] = (torch.outer(error, params['A0'])).to(device)

        return change_w

    def update_weights(self, change_w, device):
        for key, val in change_w.items():
            self.params[key] = ((self.params[key]).to(device) - (self.lr * val).to(device)).to(device)

    def compute_accuracy(self, test_data, test_y_data, batch_size):
        predictions = []
        for i in range(0, len(test_data), batch_size):
            test_set = test_data[i:i+batch_size].to(device)
            test_y_set = test_y_data[i:i + batch_size].to(device)
            for x in range(len(test_set)):
                inputs = (test_set[x] / 255.0 * 0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(test_y_set[x])] = 0.99
                output = self.forward_propagation(inputs, device)
                pred = torch.argmax(output)
                if pred == torch.argmax(torch.tensor(targets)):
                    predictions.append(1)
                else:
                    predictions.append(0)
        return torch.mean(torch.tensor(predictions), dtype=torch.float32)

    def train(self, train_list, y_list, test_list, test_y_list, device):
        batch_size = 500
        eval_batch_size = 500
        for i in range(self.epochs):
            train_predictions = []

            for i_b in range(0, train_list.shape[0], batch_size):
                train_set = train_list[i_b : (i_b + batch_size)].to(device)
                y_set = y_list[i_b : (i_b + batch_size)].to(device)

                for x in range(len(train_set)):
                    inputs = (train_set[x] / 255.0 * 0.99) + 0.01
                    targets = torch.zeros(10) + 0.01
                    targets[int(y_set[x])] = 0.99
                    output = self.forward_propagation(inputs, device)

                    pred = torch.argmax(output)
                    target_tens = targets.clone().detach()
                    if pred == torch.argmax(target_tens):
                        train_predictions.append(1)
                    else:
                        train_predictions.append(0)

                    change_w = self.backward_pass(targets, output, device)
                    self.update_weights(change_w, device)

            train_acc = torch.mean(torch.tensor(train_predictions), dtype=torch.float32)
            accuracy = self.compute_accuracy(test_list, test_y_list, eval_batch_size)
            print('Epoch: {0}, Train Accuracy: {1:.2f}%, Test Accuracy: {2:.2f}%'.format(i + 1,
                train_acc * 100, accuracy * 100))


if __name__ == '__main__':
    # Specify the location where the dataset will be stored (depends if code will be run locally or using Google Colab)
    # Case 1 - Locally
    try:
        dir_to_download_to = Path(__file__).parent  # In this case, the location will be the same as for the python file
        print('Code is executed in your local machine')
    # Case 2 - Google Colab
    except NameError as error:
        dir_to_download_to = '/'
        print('Code is executed in Google Colab')

    # Train Data
    train_data = datasets.MNIST(
        root=dir_to_download_to, train=True, download=True
    )

    # Test Data
    test_data = datasets.MNIST(
        root=dir_to_download_to, train=False, download=True
    )

    training_dataset = []
    training_labels = []
    for image, label in train_data:
        tensor = np.array(image).flatten()
        training_dataset.append(tensor)
        training_labels.append(label)

    test_dataset = []
    test_labels = []
    for image, label in test_data:
        tensor = np.array(image).flatten()
        test_dataset.append(tensor)
        test_labels.append(label)

    # ------------------------------ TRAINING DATA ------------------------------
    # This will be the input for the network
    x_inputs = torch.tensor(np.array(training_dataset), dtype=float)
    # Create the tensor which contains the true labels
    y_true = torch.tensor(training_labels)
    # ------------------------------ # ------------------------------

    # ------------------------------ TEST DATA ------------------------------
    # This will be the test input for the network
    test_x_inputs = torch.tensor(np.array(test_dataset), dtype=float)
    # Create the tensor which contains the true labels for test instances
    test_y_true = torch.tensor(test_labels)
    # ------------------------------ # ------------------------------

    '''
    print("CUDA GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        x_inputs = x_inputs.to("cuda:0")
        y_true = y_true.to("cuda:0")
        test_x_inputs = test_x_inputs.to("cuda:0")
        test_y_true = test_y_true.to("cuda:0")
        device = "cuda:0"
    else:
    '''
    device = "cpu"

    my_network = Network([784, 100, 10], epochs=25, lr=0.001)
    my_network.train(x_inputs, y_true, test_x_inputs, test_y_true, device)

