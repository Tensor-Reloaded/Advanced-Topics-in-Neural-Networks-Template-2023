import Activations
from Network import Network

def main():
    network = Network("./mnist.pkl.gz")
    
    network.add_layer(784, 100, Activations.sigmoid)
    network.add_layer(100, 10, Activations.softmax)

    network.train(network.training_set, network.testing_set)
    network.test_accuracy(network.testing_set)


if __name__ == '__main__':
    main()