# Homework 6 - Convolutional neural networks

## Abstract

This solution contains a Neural-Network model which uses Convolutions in order to achieve a higher accuracy.

## Setup

The model requires only pytorch, pytorch-torchvision. Below one can find particularities for specific accelerators.

### Intel setup

In order to setup the model to run on Intel Xe or Intel Arc, one must:
1. follow the guide on [how to setup Intel Extension for Pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.0.110%2Bxpu)
1. build a virtual environment using `$ python -m venv ./venv`
1. install the requirements from `requirements.intel.txt` (or the ones from the guide above) 
1. run `$ source setup_intel.sh`

## Running

In order to train the model, the following command can be used:

```sh
$ ./run.sh
```

## Model

The model contains:
1. 3 Convolutional layers
1. 3 Fully-connected layers

Each convolutional layer goes through a MaxPooling2D pooling function.

## Dataset and dataset transformations

The dataset used for this model is the CIFAR-10 dataset.

Training data entries undergo the following transformations:
1. Transformation to image tensor
1. Transformation to `float32` data type
<!-- 1. Random resized crop
1. Random horizontal flipping -->
1. Normalisation

The labels of the dataset are transformed using OneHot.

## Results

### Outputs

While running, I collected the following data:

| Run | Epochs | Training loss | Training accuracy | Validation loss | Validation accuracy |
| - | - | - | - | - | - |
| 1 | 25 | 206.96 | 91.77% | 189.87 | 41.98% |

### Timings

The model underwent the following:
* compilation
* tracing
* scripting

With the following results:

| Process | Epochs | Runs | Best timing | Worst timing |
| - | - | - | - | - |
| Base model | 25 | 0 | x | y |
| Compiled model | 25 | 0 | x | y |
| Traced model | 25 | 0 | x | y |
| Scripted model | 25 | 0 | x | y |