import torch
import torch.nn as nn

# Define a function for the feedforward pass through the network
def feed_forward(input, weights, biases):
    # Transpose the weights matrix to match dimensions
    weights_t = torch.transpose(weights, 0, 1)
    
    # Perform a matrix multiplication and add biases
    z = torch.matmul(weights_t, input) + biases
    
    # Apply the softmax activation function along the 0th dimension (column-wise)
    return nn.Softmax(dim=0)(z)

# Define a function to compute gradients for weights and biases
def compute_gradients(input, predicted_values, labels):
    # Compute the gradient for weights using element-wise multiplication
    gradient_w = torch.mul(predicted_values - labels, input)
    
    # Compute the gradient for biases
    gradient_b = predicted_values - labels
    
    return gradient_w, gradient_b

if __name__ == "__main__":
    LEARNING_RATE = 1
    
    # Define input data, weights, biases, and labels
    x = torch.tensor([1.0, 3.0, 0])
    weights = torch.tensor([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
    biases = torch.tensor([0.1, 0.1, 0.1])
    labels = torch.tensor([0, 1.0, 0])

    # Perform the feedforward pass to get predicted values
    predicted_values = feed_forward(x, weights, biases)

    # Print the predicted values
    print(f"Predicted values: {predicted_values.tolist()}")

    # Compute gradients for weights and biases
    delta_weights, delta_biases = compute_gradients(x, predicted_values, labels)
    
    # Print the computed gradients
    print(f"Delta weights: {delta_weights.tolist()}")
    print(f"Delta biases: {delta_biases.tolist()}")

    # Update weights and biases using gradient descent
    weights -= delta_weights * LEARNING_RATE
    biases -= delta_biases * LEARNING_RATE

    # Print the adjusted weights and biases
    print(f"Adjusted weights: {weights}")
    print(f"Adjusted biases: {biases}")
