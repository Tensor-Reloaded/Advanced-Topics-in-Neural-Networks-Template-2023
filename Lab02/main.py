import torch
import idx2numpy
import numpy as np
from torch import Tensor

def sigmoid(z: Tensor) -> Tensor:
  return 1 / (1 + torch.exp(-z))

def convert_labels(x: Tensor, size: int) -> Tensor:
  result = torch.zeros((size, 10))
  for i in range(size):
    result[i][x[i].int()] = 1
  return result

def accuracy(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor):
  count = 0
  for i in range(10000):
    # Check if each instance is classified correctly
    z = W.T @ X[i] + b
    y = sigmoid(z)
    y = torch.tensor([1 if element > 0 else 0 for element in y])
    equal = True
    for j in range(10):
      if y[j] != y_true[i][j]:
        equal = False
        break
    if equal:
        count += 1
  print("Accuracy: ", count * 100 / 10000)

def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
  # Return the updated W and b
  for i in range(10):
    z = W[:,i] @ X.T + b[i]
    y = sigmoid(z)
    y = torch.tensor([1 if element > 0 else 0 for element in y])
    error = y_true[:,i] - y
    W[:,i] = W[:,i] + mu * (X.T @ error)
    b[i] = b[i] + mu * error.mean()
  return (W, b)

train_image_file_path = "MNIST-Dataset/train-images.idx3-ubyte"
train_label_file_path = "MNIST-Dataset/train-labels.idx1-ubyte"
test_image_file_path = "MNIST-Dataset/t10k-images.idx3-ubyte"
test_label_file_path = "MNIST-Dataset/t10k-labels.idx1-ubyte"

train_images = idx2numpy.convert_from_file(train_image_file_path)
train_images = train_images.reshape(60000, 784)

# The next line was found on https://www.kaggle.com/code/prabhatverma18/mnist-gan
# It is used to normalize the values to [-1, 1]
train_images = (train_images - 127.5) / 127.5
train_images = torch.tensor(train_images, dtype = torch.float)
train_labels = idx2numpy.convert_from_file(train_label_file_path)
train_labels = torch.tensor(train_labels, dtype = torch.float)
train_labels = convert_labels(train_labels, 60000)

test_images = idx2numpy.convert_from_file(test_image_file_path)
test_images = test_images.reshape(10000, 784)
test_images = (test_images - 127.5) / 127.5
test_images = torch.tensor(test_images, dtype = torch.float)
test_labels = idx2numpy.convert_from_file(test_label_file_path)
test_labels = torch.tensor(test_labels, dtype = torch.float)
test_labels = convert_labels(test_labels, 10000)

W = torch.rand(784, 10)
b = torch.rand(10)
mu = 0.1

# Check accuracy without training
accuracy(test_images, W, b, test_labels)
for i in range(150):
    print("Epoch: ", i)
    W, b = train_perceptron(train_images, W, b, train_labels, mu)
    if i % 10 == 0:
      # Check accuracy every 10-th time
      accuracy(test_images, W, b, test_labels)
accuracy(test_images, W, b, test_labels)

# On a run I did with 150 epochs and learning_rate = 0.1, the results were the following:
# Before train: 0% accuracy
# After 50 epochs: 56.18% accuracy
# After 100 epochs: 72.12% accuracy
# After 150 epochs: 68.91% accuracy
# Best accuracy: 72.34% on epoch 102