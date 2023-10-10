import numpy as np

if __name__ == '__main__':
    # Given data
    x = np.array([1, 3, 0])
    W = np.array([[0.3, 0.1, -2],
                  [-0.6, -0.5, 2],
                  [-1, -0.5, 0.1]])
    b = np.array([0.1, 0.1, 0.1])
    y = np.array([0, 1, 0])

    print('Original W = ', W, '\n')
    print('Original b = ', b, '\n')

    # 1. Begin by computing the linear combination z for each class:
    # z = transposed(W)x + b
    z = np.dot(W.transpose(), x) + b
    print('z = ', z, '\n')

    # 2. Apply the softmax function to get the predicted probabilities y_hat:
    y_hat = [np.exp(i) / (sum(list(map(np.exp, z)))) for i in z]
    print('y_hat = ', np.round(y_hat, decimals=4), '\n')

    # 3. Compute the gradients of the loss with respect to z using the cross-entropy loss and the true labels y:
    nabla_zl = y_hat - y
    print('nabla_zl = ', np.round(nabla_zl, decimals=4), '\n')

    # 4. Now, compute the gradients with respect to the weights W and biases b:
    nabla_w_l = [np.array(np.array(nabla_zl[i] * x.transpose())) for i in range(len(z))]
    nabla_w_l = np.stack(nabla_w_l, axis=0)
    nabla_b_l = nabla_zl

    print('nabla_w_l = ', np.round(nabla_w_l, decimals=4), '\n')
    print('nabla_b_l = ', np.round(nabla_b_l, decimals=4), '\n')

    # 5. Finally, update the weights and biases using a learning rate:
    lr = 0.1
    W = np.round(W - lr*nabla_w_l, decimals=2)
    b = np.round(b - lr*nabla_b_l, decimals=4)

    print('Updated W = ', W, '\n')
    print('Updated b = ', b, '\n')