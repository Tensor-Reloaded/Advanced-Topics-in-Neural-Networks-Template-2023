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

In order to run the model using preexistent weights and biases, the following command can be used:

```sh
$ ./run_inference.sh
```

## Model

The model contains:
1. 3 Convolutional layers
1. 1 Max pooling layer
1. 2 Dropout layers
1. 3 Fully-connected layers

Each convolutional layer goes through a MaxPooling2D pooling function.

## Dataset and dataset transformations

The dataset used for this model is the CIFAR-10 dataset.

Training data entries undergo the following transformations:
1. Transformation to image tensor
1. Transformation to `float32` data type
1. Normalisation

The labels of the dataset are transformed using OneHot.

## Results

### Outputs

While running, I collected the following data:

| Run | Epochs | Training loss | Training accuracy | Validation loss | Validation accuracy |
| - | - | - | - | - | - |
| 1 | 25 | 206.96 | 91.77% | 189.87 | 41.98% |
| 2 | 25 | 144.19 | 94.15% | 43.45 | 77.45% |
| 3 | 25 | 93.38 | 96.60% | 74.14 | 75.66% |
| 4 | 25 | 80.34 | 97.45% | 72.02 | 78.24% |
| 5 | 25 | 80.16 | 97.54% | 74.10 | 77.82% |
| 6 | 25 | 131.40 | 94.84% | 43.26 | 77.21% |
| 7 | 25 | 91.84 | 96.87% | 84.95 | 74.31% |
| 8 | 25 | 78.22 | 97.60% | 74.69 | 77.84% |
| 9 | 25 | 78.95 | 97.99% | 89.58 | 77.69% |
| 10 | 25 | 109.92 | 95.92% | 54.21 | 77.36% |
| 11 | 25 | 93.39 | 96.81% | 68.61 | 77.12% |
| 12 | 25 | 83.16 | 97.50% | 118.63 | 74.66% |
| 13 | 25 | 84.18 | 97.63% | 96.15 | 76.83% |
| 14 | 25 | 83.16 | 97.50% | 118.63 | 74.66% |
| 15 | 25 | 159.88 | 93.25% | 42.02 | 76.58% |

Using Tensorboard, the following graph was generated

![Validation accuracy graph](docs/resources/validation_accuracy_graph.png)


### Timings

The model underwent the following:
* compilation
* tracing
* scripting

With the following results:

| Process | Epochs | Runs | Best timing | Worst timing |
| - | - | - | - | - |
| Base model (`CPU`) | 25 | 2 | 2324s | 2413s |
| Base model (`CUDA`) | 25 | 3 | 150.3s | 151.1s |
| Compiled model (`CUDA`) | 25 | 3 | 160.3s | 165.9s |
| Traced model (`CUDA`) | 25 | 3 | 144.5s | 148.8s |
| Scripted model (`CUDA`) | 25 | 3 | 145.1s | 149.8s |

**LEGEND**: `CPU` = My laptop's Intel Core i7 1260p, `CUDA` = A rented server with 1 x RTX A5000

**NOTE**: only the forward function was compiled/traced/scripted.

## Grading

I consider I have performed the following in this project:

| Point | Point name | Grading | My grading | Motivation
| - | - | - | - | - |
| * | Implement the Conv2d layer by hand using tensor operations | 4p | 4p | I feel like I have implemented this. |
| 1 | Try to compile, trace and script your model. Check the runtime performance and compare it to the base model | 1p | 1p | I have covered this in a section above. |
| 2 | Achieve 94% validation accuracy on CIFAR-10 | 2p | 0p | I did not achieved this |
| 3 | Achieve 95.5% validation accuracy on CIFAR-19 | 2p | 0p | I did not achieve this |
| 4 | **Bonus**: Use transfer learning to finetune a pretrained model on CIFAR-10. Achieve 97% validation accuracy on CIFAR-10 using a finetuned model | 1p | 0p | I did not achieve this |
| 5 | **Bonus**: Achieve 97% validation accuracy on CIFAR-10, without using a pretrained model | 2p | 0p | I did not achieve this |
| 6 | **Bonus**: Change the dataset to CIFAR-100. Achieve over 75% validation accuracy on CIFAR-100 without using a pretrained model | 2p | 0p | I did not achieve this |
| 7 | **Bonus**: Also test a pretrained model on CIFAR-100 and compare it with the previous results | 1p | 0p | I did not achieve this |
| * | Add a README file | 0p | 0p | - |
| * | Explanation of all layers of the model in the LaTeX document | -2p | 0p | I have detailed my model in the document |
| * | Explanation of the model's forward function in the LaTeX document | -2p | 0p | I have detailed the forward function in the document |
| * | Explanation of the model's gradient flow in the LaTeX document | -2p | 0p | I have detailed the gradient flow in the document |
| * | TensorBoard files or weights & biases link | 0p | 0p | I have this in this README.md file |
| * | PNG file with a plot of the validation accuracy of the model | 0p | 0p | I have this in this README.md file |
| * | The `inference.py` file | 0p | 0p | The file should be available under `/src` |
| - | Total | 15p | 5p | I think this is the grade I really achieved |
