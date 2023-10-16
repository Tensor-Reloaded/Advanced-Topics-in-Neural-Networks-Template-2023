import pandas
import torch
from torch import tensor
from torch.utils.data import TensorDataset, Subset, random_split
from torch.nn.functional import one_hot

from multiclass_logistic_regression import MulticlassLogisticRegression

def readDataSet() -> TensorDataset:
    csvData = pandas.read_csv("winequality-red.csv")
    features = tensor(csvData.drop('quality', axis=1).values, dtype=torch.float)
    labels = tensor(csvData['quality'].values, dtype=torch.long)
    labelsOneHot = one_hot(labels, num_classes=11)
    dataset = TensorDataset(features, labelsOneHot)

    return dataset

def splitDataSet(dataset: TensorDataset) -> tuple[Subset, Subset]:
    trainingDataLength = int(0.8 * len(dataset))
    testDataLength = len(dataset) - trainingDataLength
    trainingData, testData = random_split(dataset, [trainingDataLength, testDataLength])

    return trainingData, testData

def main():
    dataset = readDataSet()
    trainingData, testData = splitDataSet(dataset)

    neuron = MulticlassLogisticRegression(W=torch.randn(11, 11), b=torch.zeros(11), eta=0.5)

    neuron.train(trainingData, 1000)

    trainingDataAccuracy = neuron.getAccuracy(trainingData)
    testDataAccuracy = neuron.getAccuracy(testData)

    print(f"Training data accuracy: {trainingDataAccuracy}")
    print(f"Test (new) data accuracy: {testDataAccuracy}")

    # tends to be accurate @ 40% of the time for both training data and new data.
    # doing the training more times (more epochs) seems to do jack sh*t.

if __name__ == "__main__":
    main()


    