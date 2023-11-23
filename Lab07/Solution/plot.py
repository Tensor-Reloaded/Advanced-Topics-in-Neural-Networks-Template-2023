import matplotlib.pyplot as plt
import csv


def plot():
    # "RMSprop", "Adagrad", "SAM with SGD"
    # "RMSprop", "Adagrad", "SAM with SGD"

    opt_names = ["SGD", "Adam", "RMSprop", "Adagrad", "SAM with SGD"]

    no_epochs = 50
    epochs = [i for i in range(1, no_epochs + 1)]

    for name in opt_names:
        file_name = "data/" + name + ".csv"
        val_accuracy = []

        with open(file_name, mode='r') as file:
            data = csv.reader(file)

            for index, line in enumerate(data):
                if index != 0:
                    val_accuracy.append(float(line[2]))

        plt.plot(epochs, val_accuracy, label=name)

    plt.xlim([0, 50])
    plt.ylim([0, 0.6])
    plt.xlabel("No Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Performance of 5 models with different optimizers")

    plt.legend()
    plt.show()


plot()
