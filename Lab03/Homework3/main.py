import torch
from torch import Tensor
from typing import Callable

from torchvision import datasets, transforms

import MLP
from torch.utils.data import DataLoader

import util

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
#
# # Define batch size
# batch_size = 100
#
# # Create data loaders for the datasets
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # Split data into x_train, y_train, x_test, y_test
# x_train, y_train = next(iter(train_loader))
# x_test, y_test = next(iter(test_loader))
#
# # Ensure x_train, x_test have the shape [batch_size, num_features]
# x_train = x_train.view(x_train.size(0), -1)
# x_test = x_test.view(x_test.size(0), -1)
#
# y_train = util.convert_to_one_hot(y_train)
# y_test = util.convert_to_one_hot(y_test)

torch.device('cpu')

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

x_train = mnist_trainset.data
x_train = torch.flatten(x_train, start_dim=1)
y_train_init = mnist_trainset.targets
y_train = util.convert_to_one_hot(y_train_init)

x_train = x_train.float()
y_train = y_train.float()

x_train = x_train / 255

x_test = mnist_testset.data
x_test = torch.flatten(x_test, start_dim=1)
y_test_init = mnist_testset.targets
y_test = util.convert_to_one_hot(y_test_init)

x_test = x_test.float()
y_test = y_test.float()

x_test = x_test / 255

if __name__ == "__main__":
    model = MLP.MLP([784, 100, 10])
    model.train(x_train, y_train, batch_size=100, lr=0.01, epochs=100)

    # Test model
    acc = model.test(x_test, y_test)
    print(f"Accuracy: {acc}")

# Epoch 0 done
# Loss: 1.5743049383163452
# Accuracy:  0.9289166666666666
# ---------------------------------------
# Epoch 10 done
# Loss: 1.4864616394042969
# Accuracy:  0.9892
# ---------------------------------------
# Epoch 20 done
# Loss: 1.4736779928207397
# Accuracy:  0.9968833333333333
# ---------------------------------------
# Epoch 30 done
# Loss: 1.4681593179702759
# Accuracy:  0.9994833333333333
# ---------------------------------------
# Epoch 40 done
# Loss: 1.465474247932434
# Accuracy:  0.9998666666666667
# ---------------------------------------
# Epoch 50 done
# Loss: 1.464221477508545
# Accuracy:  0.99995
# ---------------------------------------
# Epoch 60 done
# Loss: 1.463523268699646
# Accuracy:  0.9999833333333333
# ---------------------------------------
# Epoch 70 done
# Loss: 1.463023066520691
# Accuracy:  0.9999833333333333
# ---------------------------------------
# Epoch 80 done
# Loss: 1.462689757347107
# Accuracy:  0.9999833333333333
# ---------------------------------------
# Epoch 90 done
# Loss: 1.4624584913253784
# Accuracy:  1.0
# ---------------------------------------
# Accuracy: 0.9804
