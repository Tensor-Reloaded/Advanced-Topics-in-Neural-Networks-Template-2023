import numpy as np
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


x = np.array([1, 3, 0])
W = np.array([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
b = np.array([0.1, 0.1, 0.1])
y = np.array([0, 1, 0])
learning_rate = 0.1

z = np.matmul(W.transpose(), x) + b
print(z)
predicted_y = softmax(z)
print(predicted_y)

delta_z = predicted_y - y
print(delta_z)
delta_W = np.matmul(delta_z, x.transpose())
delta_b = delta_z

W = W - learning_rate * delta_W
b = b - learning_rate * delta_b
print(W, b)

df = pd.read_csv('WineQT.csv')
df = df.drop('Id', axis = 1)


def start_training(df):
    train = df.sample(frac=0.6, random_state=200)
    validate = df.drop(train.index).sample(frac=0.5)
    test = df.drop(train.index).drop(validate.index)
    W = np.random.randint(-2, 2, size=(11, 11))
    b = np.array([0.1 for i in range(11)])

    for epoch in range(500):
        print('The number of epoch is', epoch)
        for index, line in train.iterrows():
            x = line.values
            x = [float(value) for value in x]
            y = np.array([1 if i == x[-1] else 0 for i in range(0, 11)])
            x = np.array(x[:-1])

            z = np.matmul(W.transpose(), x) + b
            predicted_y = softmax(z)

            delta_z = predicted_y - y
            delta_W = np.matmul(delta_z, x.transpose())
            delta_b = delta_z

            W = W - learning_rate * delta_W
            b = b - learning_rate * delta_b

        count_correct = 0
        for index, line in validate.iterrows():
            x = line.values
            x = [float(value) for value in x]
            y = x[-1]
            x = np.array(x[:-1])

            z = np.matmul(W.transpose(), x) + b
            predicted_y = softmax(z)
            result = np.argmax(predicted_y)

            if result == y:
                count_correct += 1

        print(count_correct / len(validate.index))

    count_correct = 0
    for index, line in test.iterrows():
        x = line.values
        x = [float(value) for value in x]
        y = x[-1]
        x = np.array(x[:-1])

        z = np.matmul(W.transpose(), x) + b
        predicted_y = softmax(z)
        result = np.argmax(predicted_y)

        if result == y:
            count_correct += 1

    print(count_correct / len(test.index))

    with open("weights1.txt", "w") as txt_file:
        txt_file.write(str(W))

    with open("bais1.txt", "w") as txt_file:
        txt_file.write(str(b))


learning_rate = 0.15
start_training(df)