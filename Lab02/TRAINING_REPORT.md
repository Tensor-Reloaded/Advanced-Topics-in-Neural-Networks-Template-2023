# Neural Network Training Report

## Dataset

The model was trained on the MNIST dataset, which consists of handwritten digits.

## Training Results

| Metric           | Value     |
|------------------|-----------|
| Initial Accuracy | 9.83% |
| Final Accuracy | 90.61% |

## Model Architecture

The model used in this training process is a simple feedforward neural network with a single hidden layer.
The model architecture is as follows:

1. Input Layer: 784 neurons (corresponding to 28x28 pixel images)
2. Hidden Layer: 10 neurons with sigmoid activation
3. Output Layer: 10 neurons (corresponding to digit classes)

## Training Details

The model was trained with the following hyperparameters:

- Learning Rate: 0.01
- Maximum Training Iterations: 10

## Conclusion

The neural network was trained on the MNIST dataset and achieved the following results:

- Initial Accuracy: 9.83%
- Final Accuracy: 90.61%

Further optimization and fine-tuning of hyperparameters may lead to improved results.
