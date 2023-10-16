import gzip
import torch
import pickle

LEARNING_RATE = 0.01
MAX_ITERATIONS = 10

class FeedForwardNeuralNetwork:
    def __init__(self, dataset_file_path: str):
        self.biases = torch.rand(10)
        self.weights = torch.rand(784, 10)

        self.training_set = None
        self.validation_set = None
        self.testing_set = None
        self.load_dataset(dataset_file_path)

    def load_dataset(self, dataset_file_path: str):
        with gzip.open(dataset_file_path, "rb") as file:
            training_data, validation_data, testing_data = pickle.load(file, encoding='latin')

        self.training_set = [(torch.tensor(inputs, dtype=torch.float32), label) for inputs, label in zip(training_data[0], training_data[1])]
        self.validation_set = [(torch.tensor(inputs, dtype=torch.float32), label) for inputs, label in zip(validation_data[0], validation_data[1])]
        self.testing_set = [(torch.tensor(inputs, dtype=torch.float32), label) for inputs, label in zip(testing_data[0], testing_data[1])]

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def train(self, training_data):
        iterations = MAX_ITERATIONS
        all_classified = False

        while not all_classified and iterations > 0:
            all_classified = True
            for inputs, correct_label in training_data:
                expected_output = torch.tensor([1 if i == correct_label else 0 for i in range(10)])

                output = torch.matmul(inputs, self.weights) + self.biases
                activated_output = torch.tensor([self.sigmoid(value) for value in output])

                self.weights += torch.matmul(inputs.view(784, 1), (expected_output - activated_output).view(1, 10)) * LEARNING_RATE
                self.biases += (expected_output - activated_output) * LEARNING_RATE

                if not torch.equal(activated_output, expected_output):
                    all_classified = False

            iterations -= 1

    def predict(self, inputs):
        output = torch.matmul(inputs, self.weights) + self.biases
        activated_output = torch.tensor([self.sigmoid(value) for value in output])
        return torch.argmax(activated_output)

    def test_model(self, test_data):
        correct_predictions = 0
        total = 0

        for inputs, target_label in test_data:
            predicted_label = self.predict(inputs)
            if predicted_label == target_label:
                correct_predictions += 1

            total += 1

        accuracy = correct_predictions / total
        return accuracy

def main():
    model = FeedForwardNeuralNetwork("./mnist.pkl.gz")

    training_results = []

    initial_accuracy = model.test_model(model.testing_set)
    training_results.append(("Initial Accuracy", initial_accuracy))

    model.train(model.training_set)
    model.train(model.validation_set)

    final_accuracy = model.test_model(model.testing_set)
    training_results.append(("Final Accuracy", final_accuracy))

    with open("TRAINING_REPORT.md", "w") as markdown_file:
        markdown_file.write("# Neural Network Training Report\n\n")
        markdown_file.write("## Dataset\n\n")
        markdown_file.write("The model was trained on the MNIST dataset, which consists of handwritten digits.\n\n")

        markdown_file.write("## Training Results\n\n")
        markdown_file.write("| Metric           | Value     |\n")
        markdown_file.write("|------------------|-----------|\n")
        for result in training_results:
            markdown_file.write(f"| {result[0]} | {result[1]:.2%} |\n")

        markdown_file.write("\n## Model Architecture\n\n")
        markdown_file.write("The model used in this training process is a simple feedforward neural network with a single hidden layer.\n")
        markdown_file.write("The model architecture is as follows:\n\n")
        markdown_file.write("1. Input Layer: 784 neurons (corresponding to 28x28 pixel images)\n")
        markdown_file.write("2. Hidden Layer: 10 neurons with sigmoid activation\n")
        markdown_file.write("3. Output Layer: 10 neurons (corresponding to digit classes)\n\n")

        markdown_file.write("## Training Details\n\n")
        markdown_file.write("The model was trained with the following hyperparameters:\n\n")
        markdown_file.write(f"- Learning Rate: {LEARNING_RATE}\n")
        markdown_file.write(f"- Maximum Training Iterations: {MAX_ITERATIONS}\n")

        markdown_file.write("\n## Conclusion\n\n")
        markdown_file.write("The neural network was trained on the MNIST dataset and achieved the following results:\n\n")
        markdown_file.write(f"- Initial Accuracy: {initial_accuracy:.2%}\n")
        markdown_file.write(f"- Final Accuracy: {final_accuracy:.2%}\n\n")
        markdown_file.write("Further optimization and fine-tuning of hyperparameters may lead to improved results.\n")
    print("Finished training")


if __name__ == '__main__':
    main()
