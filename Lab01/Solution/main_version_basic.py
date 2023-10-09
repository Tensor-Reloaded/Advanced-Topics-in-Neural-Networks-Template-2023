import numpy as np
from numpy import ndarray
from typing import Tuple


def compute_weighted_sum_features(features: np.ndarray, weights: ndarray, biases: ndarray) -> Tuple[ndarray]:
    if weights.shape[0] != features.shape[0]:
        raise AttributeError("Wrong parameters dimensions!")
    return np.array([(sum([features[i] * weights[i][j] for i in range(weights.shape[0])])
                      + biases[j]) for j in range(weights.shape[1])])
    # return np.matmul(np.transpose(w), x) + biases


def sigmoid_function(output_number: float) -> float:
    return 1.0 / (1.0 + np.exp(-output_number))


def softmax_function(output_vector: ndarray) -> Tuple[ndarray]:
    return (1.0 / sum([np.exp(output_vector[i]) for i in range(output_vector.shape[0])])
            * np.array([np.exp(output_vector[i]) for i in range(output_vector.shape[0])]))


def apply_activation_function(computed_output: ndarray, real_output: ndarray,
                              weights: ndarray, biases: ndarray, it: int = 1) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    if real_output.shape[0] == 1:  # apply sigmoid function
        if it == 1:
            prob = sigmoid_function(computed_output[0])
        else:
            prob = sigmoid_function(computed_output[1])
        if real_output[0] == 0.0:
            real_output = np.array([1.0, 0.0])  # belongs to False (0) class
            if it == 1:
                weights = (np.concatenate((weights[:, 0], weights[:, 0]))
                           .reshape(2, len(weights[:, 0]))).transpose()
                biases = biases.tolist()
                biases.extend(biases)
                biases = np.array(biases)
            return weights, biases, real_output, np.array([prob, 1.0 - prob])
        elif real_output[0] == 1.0:
            real_output = np.array([0.0, 1.0])  # belongs to True (1) class
            if it == 1:
                weights = (np.concatenate((weights[:, 0], weights[:, 0]))
                           .reshape(2, len(weights[:, 0]))).transpose()
                biases = biases.tolist()
                biases.extend(biases)
                biases = np.array(biases)
            return weights, biases, real_output, np.array([1.0 - prob, prob])
        else:
            raise ArithmeticError("Real output can only take"
                                  " binary values in this case!")
    else:  # apply softmax function
        return weights, biases, real_output, softmax_function(computed_output)


def logistic_regression_gradients_calculus(features: ndarray, weights: ndarray,
                                           biases: ndarray, real_out: ndarray) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    initial_output = compute_weighted_sum_features(features, weights, biases)
    print(f"z: {initial_output}")
    weights, biases, real_out, activated_output = apply_activation_function(
        initial_output, real_out, weights, biases)  # treated binary case as multiclass case
    print(f"y_pred: {activated_output}")
    loss = define_loss_as_cross_entropy(activated_output, real_out)
    print(f"Loss for current iteration: {loss}")
    gradient_z = activated_output - real_out
    print(f"Gradient z: {gradient_z}")
    gradient_w = np.matmul(features.reshape(len(features), 1), np.array([gradient_z]))
    print(f"Gradient w: {gradient_w}")
    gradient_b = np.copy(gradient_z)
    print(f"Gradient b: {gradient_b}")
    return weights, biases, gradient_z, gradient_w, gradient_b


def logistic_regression_update(weights: ndarray, biases: ndarray, gradient_w: ndarray,
                               gradient_b: ndarray, learning_rate: ndarray) -> Tuple[ndarray, ndarray]:
    weights -= learning_rate * gradient_w
    biases -= learning_rate * gradient_b
    return weights, biases


def define_loss_as_cross_entropy(prob_output: ndarray, real_output: ndarray) -> float:
    if prob_output.shape[0] != real_output.shape[0]:
        raise AttributeError("Not same number of output neurons!")
    return -np.sum([(real_output[i] * np.log(np.clip(prob_output[i], 1e-12, None)))
                    for i in range(len(prob_output))])


def solved_demo():
    x = np.array([1.0, 3.0, 0.0])  # think of it as column vector
    w = np.array([[-0.6, -0.5, 2.0]]).reshape(3, 1)
    b = np.array([0.1])
    y = np.array([1.0])  # real output
    miu = 0.2
    adapted_weights, adapted_biases, update_z, update_w, update_b \
        = logistic_regression_gradients_calculus(x, w, b, y)
    weights, biases = logistic_regression_update(adapted_weights, adapted_biases,
                                                 update_w, update_b, miu)
    # weights = np.transpose(weights)
    print(f"Updated weights: {weights}")
    print(f"Updated biases: {biases}")


def proposed_demo():
    x = np.array([1.0, 3.0, 0.0])  # think of it as column vector
    w = np.array([[0.3, 0.1, -2.0], [-0.6, -0.5, 2.0], [-1.0, -0.5, 0.1]]).transpose()
    b = np.array([0.1, 0.1, 0.1])
    y = np.array([0, 1, 0])  # real output
    miu = 0.2
    adapted_weights, adapted_biases, update_z, update_w, update_b \
        = logistic_regression_gradients_calculus(x, w, b, y)
    weights, biases = logistic_regression_update(adapted_weights, adapted_biases,
                                                 update_w, update_b, miu)
    # weights = np.transpose(weights)
    print(f"Updated weights: {weights}")
    print(f"Updated biases: {biases}")


if __name__ == "__main__":
    # solved_demo()
    proposed_demo()
