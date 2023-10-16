import gzip
import torch
import pickle
from tqdm import tqdm


class Model:
    LEARNING_RATE_TRAIN = 0.1
    LEARNING_RATE_VALIDATION = 0.01
    TRAINING_MAX_ITERATIONS = 10

    def __init__(self, dataset_file_path: str):
        self.biases =  torch.rand(10)  # Shape (1, 10)
        self.weights = torch.rand(784, 10)  # Shape (784, 10)
        
        self.training_set = None
        self.validation_set = None
        self.testing_set = None
        self.load_dataset(dataset_file_path)

    def load_dataset(self, dataset_file_path: str):
        with gzip.open(dataset_file_path, "rb") as fd:
            training_set, validation_set, testing_set = pickle.load(fd, encoding='latin')

        self.training_set = [(torch.tensor(inputs, dtype=torch.float32), tag) for inputs, tag in zip(training_set[0], training_set[1])]
        self.validation_set = [(torch.tensor(inputs, dtype=torch.float32),tag) for inputs, tag in zip(validation_set[0], validation_set[1])]
        self.testing_set = [(torch.tensor(inputs, dtype=torch.float32), tag) for inputs, tag in zip(testing_set[0], testing_set[1])]

        print("Successfully loaded dataset!")

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))
    
    def train(self, training_set, learning_rate: float):
        iterations = self.TRAINING_MAX_ITERATIONS
        all_classified = False

        while not all_classified and iterations > 0:
            all_classified = True
            for input_values, correct_tag in tqdm(training_set, desc=f"Epoch {self.TRAINING_MAX_ITERATIONS - iterations + 1}/{self.TRAINING_MAX_ITERATIONS}"):
                expected_result = torch.tensor([1 if i == correct_tag else 0 for i in range(10)])

                output = torch.matmul(input_values, self.weights) + self.biases
                activated_output = torch.tensor([self.sigmoid(value) for value in output])

                self.weights += torch.matmul(input_values.view(784, 1), (expected_result - activated_output).view(1, 10)) * learning_rate
                self.biases += (expected_result - activated_output) * learning_rate

                if not torch.equal(activated_output, expected_result):
                    all_classified = False
            
            iterations -= 1

    def predict(self, input_values):
        output = torch.matmul(input_values, self.weights) + self.biases
        activated_output = torch.tensor([self.sigmoid(value) for value in output])
        return torch.argmax(activated_output)

    def test_model(self, test_set):
        correct_predictions = 0
        total = 0

        for input_values, target_label in test_set:
            predicted_value = self.predict(input_values)
            if predicted_value == target_label:
                correct_predictions += 1
            
            total += 1

        print(f"Correct/Total: {correct_predictions:,}/{total:,}")
        print(f"Accuracy: {int(correct_predictions / total * 10000.) / 100}%\n")


def main():
    model = Model("./mnist.pkl.gz")

    print("\nTesting initial accuracy...")
    model.test_model(model.testing_set)

    # Train model
    model.train(model.training_set, model.LEARNING_RATE_TRAIN)
    model.train(model.validation_set, model.LEARNING_RATE_VALIDATION)

    print("\nTesting final accuracy...")
    model.test_model(model.testing_set)


if __name__ == '__main__':
    main()
