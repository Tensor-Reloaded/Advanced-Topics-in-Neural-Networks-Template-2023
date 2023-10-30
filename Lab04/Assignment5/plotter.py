from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class MetricsMemory:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.t_metrics = np.zeros((epochs, 2))
        self.metrics = np.zeros((epochs, 2))

    def update_metrics(self, epoch: int, validation_update: Tuple[float, float],
                       training_update: Tuple[float, float]):
        self.metrics[epoch][0], self.metrics[epoch][1] = validation_update
        self.t_metrics[epoch][0], self.t_metrics[epoch][1] = training_update
        print(f'Training: Epoch {epoch + 1}/{self.epochs}, '
              f'Loss: {training_update[1]}, '
              f'Accuracy: {training_update[0]}\n'
              f'Testing: Epoch {epoch + 1}/{self.epochs}, '
              f'Loss: {validation_update[1]}, '
              f'Accuracy: {validation_update[0]}')

    def draw_plot(self):
        x_axis = list(range(1, self.epochs + 1))
        plt.plot(x_axis, self.metrics[:, 0])
        plt.plot(x_axis, self.t_metrics[:, 0])
        plt.plot(x_axis, self.metrics[:, 1])
        plt.plot(x_axis, self.t_metrics[:, 1])
        plt.legend(['testing accuracy', 'training accuracy',
                    'testing loss', 'training loss'], loc="lower right")
        plt.title('Evolution of accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.show()
