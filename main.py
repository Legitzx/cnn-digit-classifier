from model import *
from mnist_loader import *

"""
Network Architecture Overview:

Input: 28x28 (784 pixels) image

Convolutional Layer #1:
 - Filter size: 5x5
 - Num Filters: 32
 - Feature Map (Output): 28 - 5 + 1 = 24x24
 
Max Pooling Layer #1:
 - Filter size: 2x2
 - Num Filters 32
 - Output: 24 / 2 = 12x12

Convolutional Layer #2:
 - Filter size: 5x5
 - Num Filters: 64
 - Feature Map (Output): 12 - 5 + 1 = 8x8

Max Pooling Layer #2:
 - Filter size: 2x2
 - Num Filters 64
 - Output: 8 / 2 = 4x4

[Flatten] -> 64 4x4 = 1024 input neurons as a column vector

Dense #1:
 - Num Neurons: 128
 - Activation: ReLU
 
Dense #2: (Output)
 - Num Neurons: 10
 - Activation: Softmax


Optimizer: Stochastic Gradient Descent (SGD)
Loss Function: Categorical Cross-Entropy
"""


def main():
    X_train, Y_train, X_test, Y_test = get_mnist_data()

    layers = [
        Convolutional((1, 28, 28), 32, 5),
        Pooling((32, 24, 24)),
        Convolutional((32, 12, 12), 64, 5),
        Pooling((64, 8, 8)),
        Flatten(),
        Dense(128, (1024, 1), "relu"),
        Dense(10, (128, 1), "softmax")
    ]

    learning_rate = 0.0001
    starting_learning_rate = learning_rate

    train_data_slider_start = 0
    train_data_slider_end = 5000

    for epoch in range(1000):
        count = 0
        num_correct = 0
        total_loss = 0
        for train_input, desired_output in zip(X_train[train_data_slider_start:train_data_slider_end], Y_train[train_data_slider_start:train_data_slider_end]):

            l_input = train_input

            for layer in layers:
                l_input = layer.forward(l_input)

            if np.argmax(l_input) == np.argmax(desired_output):
                num_correct += 1


            total_loss += categorical_cross_entropy_cost(l_input, desired_output)

            # Dense Output
            grad = layers[-1].backward(None, None, learning_rate, desired_output)
            # Dense #2
            grad = layers[-2].backward(grad, layers[-1].weights, learning_rate, None)
            # Flatten
            grad = layers[-3].backward(grad, layers[-2].weights)
            # Max Pooling
            grad = layers[-4].backward(grad)
            # Conv Layer
            grad = layers[-5].backward(grad, learning_rate)
            # Max Pooling
            grad = layers[-6].backward(grad)
            # Conv Layer
            grad = layers[-7].backward(grad, learning_rate)

            if count % 1000 == 0:
                print("Train #{}".format(count))

            count += 1


        print("Epoch #", epoch)
        print("{}/{} {:.1f}% Accuracy".format(num_correct, len(X_train[train_data_slider_start:train_data_slider_end]),
                                              (num_correct / float(len(X_train[train_data_slider_start:train_data_slider_end]))) * 100.0))
        print("Total Loss: {}".format(total_loss))
        print("Average Loss: {}".format(total_loss / num_correct))

        # learning rate decay
        print("Learning rate: {}".format(learning_rate))
        learning_rate_decay = 0.001
        learning_rate = starting_learning_rate * (1.0 / (1 + learning_rate_decay * epoch))

        train_data_slider_start += 5000
        train_data_slider_end += 5000

        if train_data_slider_end >= len(X_train):
            train_data_slider_start = 0
            train_data_slider_end = 5000

        print("<--------- TEST DATA ---------->")
        num_correct = 0
        count = 0
        for test_input, test_desired_output in zip(X_test, Y_test):
            l_input = test_input

            for layer in layers:
                l_input = layer.forward(l_input)

            if np.argmax(l_input) == np.argmax(test_desired_output):
                num_correct += 1

            if count % 1000 == 0:
                print("Test #{}".format(count))

            count += 1

        print("{}/{} {:.1f}% TEST Accuracy".format(num_correct, len(X_test),
                                              (num_correct / float(len(X_test))) * 100.0))
        print("<------------------------------>")


if __name__ == '__main__':
    main()
