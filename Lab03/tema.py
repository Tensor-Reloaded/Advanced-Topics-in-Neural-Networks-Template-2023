import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

input_size = 784
hidden_size = 100
output_size = 10

np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.001

epochs = 1000
for epoch in range(epochs):
    for images, labels in train_dataset:
        # Flatten the input image
        x = images.view(-1, 784).numpy()
        y = np.zeros((1, 10))
        y[0, labels] = 1

        #forward
        hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)

        #back
        d_output = y - output_layer_output
        error_hidden_layer = d_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * hidden_layer_output * (1 - hidden_layer_output)

        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += x.T.dot(d_hidden_layer) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        print(epoch, np.mean(np.argmax(output_layer_output, axis=1) == labels))
        

correct = 0
total = 0
for images, labels in test_dataset:
    x = images.view(-1, 784).numpy()
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    predicted = np.argmax(output_layer_output)
    total += 1
    if predicted == labels:
        correct += 1

accuracy = (correct / total) * 100

print(f'Test Accuracy: {accuracy:.2f}%')#96.94% accuracy

#loss = np.mean((y - output_layer_output) ** 2)
#print(loss)

