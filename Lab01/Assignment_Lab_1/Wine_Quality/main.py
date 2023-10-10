import numpy as np
import pandas as pd


def train(X, W, b, Y, lr, epochs):
    for ep in range(epochs):
        for index in range(len(X)):
            x = input_x[index]
            y = Y[index]

            # 1. Begin by computing the linear combination z for each class:
            # z = transposed(W)x + b
            z = np.dot(W.transpose(), x) + b
            print('z = ', z, '\n')

            # 2. Apply the softmax function to get the predicted probabilities y_hat:
            y_hat = [np.exp(i) / (sum(list(map(np.exp, z)))) for i in z]
            print('y_hat = ', np.round(y_hat, decimals=4), '\n')

            # 3. Compute the gradients of the loss with respect to z using the cross-entropy loss and the true labels y:
            # print(y_hat.shape)
            print('y: ', len(y))
            nabla_zl = y_hat - y

            print('nabla_zl = ', np.round(nabla_zl, decimals=4), '\n')

            # 4. Now, compute the gradients with respect to the weights W and biases b:
            nabla_w_l = [np.array(np.array(nabla_zl[i] * x.transpose())) for i in range(len(z))]
            nabla_w_l = np.stack(nabla_w_l, axis=0)
            nabla_b_l = nabla_zl

            print('nabla_w_l = ', np.round(nabla_w_l, decimals=4), '\n')
            print('nabla_b_l = ', np.round(nabla_b_l, decimals=4), '\n')

            # 5. Finally, update the weights and biases using a learning rate:

            W = np.round(W - lr * nabla_w_l, decimals=2)
            b = np.round(b - lr * nabla_b_l, decimals=4)

            print('Updated W = ', W, '\n')
            print('Updated b = ', b, '\n')


if __name__ == '__main__':
    # Given data
    df = pd.read_csv('winequality-red.csv')
    print(df)

    input_x = (df.iloc[:, :-1]).to_numpy()

    W = np.random.rand(11, 11)
    b = np.random.rand(11)
    labels = df.iloc[:, -1:].to_numpy()

    one_hot_labels = np.zeros((len(labels), 11), dtype=int)

    for i in range(len(labels)):
        index = int(labels[i][0])
        one_hot_labels[i][index - 1] = int(1)

    lr = 1e-7
    epochs = 5

    train(input_x, W, b, one_hot_labels, lr, epochs)
