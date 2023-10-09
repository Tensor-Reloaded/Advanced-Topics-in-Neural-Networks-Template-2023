import numpy as np
from numpy import ndarray
from typing import Tuple
import main_version_basic as mvb
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures


def draw_plot(accuracies: ndarray, epochs: int, t_accuracies: ndarray):
    x_axis = list(range(1, epochs + 1))
    plt.plot(x_axis, accuracies)
    plt.plot(x_axis, t_accuracies)
    plt.legend(['testing', 'training'], loc="lower right")
    plt.title('Evolution of accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def define_dataset() -> Tuple[ndarray, ndarray]:
    dataframe = pd.read_csv("winequality-red.csv")
    all_data = np.array([dataframe[key].tolist() for key in dataframe.keys()])
    real_outputs = all_data[-1]
    all_data = all_data[:-1]
    all_data = normalize_features(all_data)
    training_set = {'data': all_data[:, :1440], 'out': real_outputs[:1440].astype(int)}
    testing_set = {'data': all_data[:, 1440:], 'out': real_outputs[1440:].astype(int)}
    return training_set, testing_set


def normalize_features(data: ndarray) -> ndarray:
    normalized_data = []
    for set in data:
        normalized_set = (set - np.min(set)) / (np.max(set) - np.min(set))
        normalized_data.append(normalized_set)
    return np.array(normalized_data)


def regroup_as_instances(column_displayed_data: ndarray) -> ndarray:
    return np.transpose(column_displayed_data)


def one_hot_encoding(results: ndarray, no_out_neurons: int) -> ndarray:
    return np.array([[0 for _ in range(instance - 3)] + [1] +
                     [0 for _ in range(instance - 2, no_out_neurons)]
                     for instance in results])


def compute_batch_loss(activated_output: ndarray, selected_output: ndarray, batch_size: int) -> ndarray:
    return np.array([mvb.define_loss_as_cross_entropy
                     (activated_output[instance], selected_output[instance])
                     for instance in range(batch_size)]).mean()


def one_run_batch(data: ndarray, out: ndarray, weights: ndarray,
                  biases: ndarray, iteration: int, batch_size: int = 16) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    # select batch_size instances
    instances = np.random.choice(range(len(data)), size=batch_size, replace=False)
    selected_data = np.array([data[instance] for instance in instances])
    selected_outputs = np.array([out[instance] for instance in instances])
    all_outputs = []
    selected_outs = []
    if iteration == 1 and len(selected_outputs[0]) == 1:
        adapted_weights = []
        adapted_biases = []
    for index, instance in enumerate(selected_data):
        initial_output = mvb.compute_weighted_sum_features(instance, weights, biases)
        adapted_weights, adapted_biases, adapted_real_out, activated_output = \
            mvb.apply_activation_function(initial_output,
                                          selected_outputs[index],
                                          weights, biases, it=iteration)
        selected_outs.append(np.copy(adapted_real_out))
        all_outputs.append(activated_output)
    all_outputs = np.array(all_outputs)
    # print(all_outputs)
    if iteration == 1 and len(selected_outputs[0]) == 1:
        weights = np.copy(adapted_weights)
        biases = np.copy(adapted_biases)
    selected_outputs = np.array(selected_outs)
    loss = compute_batch_loss(all_outputs, selected_outputs, batch_size)
    print(f"Loss for iteration {iteration}: {loss}")
    difference = (all_outputs - selected_outputs)
    gradient_w = (np.matmul(selected_data.transpose(), difference)).mean(axis=0)
    gradient_b = difference.mean()
    return weights, biases, gradient_w, gradient_b


# parallelism
# def one_run_batch(data, out, weights, biases, iteration, batch_size=8):
#     # select batch_size instances
#     instances = np.random.choice(range(len(data)), size=batch_size, replace=False)
#     selected_data = np.array([data[instance] for instance in instances])
#     selected_outputs = np.array([out[instance] for instance in instances])
#     all_outputs = []
#     selected_outs = []
#     processes = []
#     with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#         for index, instance in enumerate(selected_data):
#             initial_output = mvb.compute_weighted_sum_features(instance, weights, biases)
#             processes.append(executor.submit(mvb.apply_activation_function, initial_output,
#                                              selected_outputs[index],
#                                              weights, biases, it=iteration))
#         for p in processes:
#             selected_outs.append(np.copy(p.result()[2]))
#             all_outputs.append(p.result()[3])
#         adapted_weights = processes[0].result()[0]
#         adapted_biases = processes[0].result()[1]
#         executor.shutdown()
#
#     all_outputs = np.array(all_outputs)
#     # print(all_outputs)
#     if iteration == 1 and len(selected_outputs[0]) == 1:
#         weights = np.copy(adapted_weights)
#         biases = np.copy(adapted_biases)
#     selected_outputs = np.array(selected_outs)
#     loss = compute_batch_loss(all_outputs, selected_outputs, batch_size)
#     print(f"Loss for iteration {iteration}: {loss}")
#     difference = (all_outputs - selected_outputs)
#     gradient_w = (np.matmul(selected_data.transpose(), difference)).mean(axis=0)
#     gradient_b = difference.mean()
#     return weights, biases, gradient_w, gradient_b


def train(model: dict, training_data: dict, testing_data: dict, epochs: int,
          miu: float = 0.1, batch_size: int = 32) -> dict:
    accuracies = []
    t_accuracies = []
    for epoch in range(epochs):
        weights, biases, gradient_w, gradient_b = \
            one_run_batch(training_data['data'], training_data['out'],
                          model['weights'], model['biases'], epoch + 1, batch_size)
        model['weights'], model['biases'] = mvb.logistic_regression_update(
            weights, biases, gradient_w, gradient_b, miu)
        accuracy = test(model, testing_data)
        accuracies += [accuracy]
        accuracy = test(model, training_data)
        t_accuracies += [accuracy]
    draw_plot(accuracies, epochs, t_accuracies)
    return model


def test(model: dict, testing_data: dict) -> float:
    correct = 0
    for index, instance in enumerate(testing_data['data']):
        initial_output = mvb.compute_weighted_sum_features(instance, model['weights'], model['biases'])
        adapted_weights, adapted_biases, adapted_real_out, activated_output = \
            mvb.apply_activation_function(initial_output,
                                          testing_data['out'][index],
                                          model['weights'], model['biases'])
        maxi_index = activated_output.tolist().index(np.max(activated_output))
        if maxi_index == testing_data['out'][index].tolist().index(1):
            correct += 1
    return correct / len(testing_data['data'])


if __name__ == "__main__":
    train_data, test_data = define_dataset()
    model_i = {'no_input': 11,  # number of input features
               'weights': np.random.rand(6, len(train_data['data'])),
               'biases': np.random.rand(6),
               'no_output': len(set(test_data['out']))  # number of output scores (3-8)
               }
    train_data['data'] = regroup_as_instances(train_data['data'])
    test_data['data'] = regroup_as_instances(test_data['data'])
    train_data['out'] = one_hot_encoding(train_data['out'], model_i['no_output'])
    test_data['out'] = one_hot_encoding(test_data['out'], model_i['no_output'])

    # for assignment data
    # train_data['data'] = np.array([[1.0, 3.0, 0.0]])
    # train_data['out'] = np.array([[0, 1, 0]])
    # model_i = {'no_input': 3,  # number of input features
    #            'weights': np.array([[0.3, 0.1, -2.0], [-0.6, -0.5, 2.0], [-1.0, -0.5, 0.1]]).transpose(),
    #            'biases': np.array([0.1, 0.1, 0.1]),
    #            'no_output': 3
    #            }
    # train_data['data'] = np.array([[1.0, 3.0, 0.0]])
    # train_data['out'] = np.array([[1]])
    # model_i = {'no_input': 3,  # number of input features
    #            'weights': np.array([[-0.6, -0.5, 2.0]]).reshape(3, 1),
    #            'biases': np.array([0.1]),
    #            'no_output': 2
    #            }
    model_f = train(model_i, train_data, test_data, 1000, miu=0.02, batch_size=8)
    print(model_f)
