import numpy as np
from scipy import signal

"""
Activation Function for hidden layers: Rectified Linear Unit (ReLU)
Activation Function for output layer: Softmax
Cost function: Categorical Cross-Entropy
"""


class Convolutional:
    def __init__(self, input_shape, num_filters, filter_size):
        # architecture
        self.num_filters = num_filters

        # input
        input_depth, input_height, input_width = input_shape

        """
        Filter
        Notes:
         - Filter depth must match input depth.
        """
        he_scale = np.sqrt(2.0 / (input_depth * filter_size * filter_size))
        self.filters = np.random.randn(num_filters, input_depth, filter_size,
                                       filter_size) * he_scale

        # feature maps (output)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)

        """
        Biases:
        Notes:
         - Each term in the array represents the bias for each filter.
         - One bias per filter.
        """
        self.biases = np.zeros(num_filters)

    def forward(self, input):
        # we need to go through self.layers and convolve each filter with the input & get self.layers feature maps
        self.input = input

        """
        The input can be either 2D or 3D. SciPy does not offer a convolve function for operating on 3D matrices.
        Lets assume the input to this layer is 3D. We will first create a loop that goes through each filter
        in the layer. Then within that loop, we will have another that loops through each DEPTH in the input. Now 
        we can operate on a 2D slice of the input and a 2D slice of the filter, since the input and filter must have
        the same depth.
        """
        output = np.zeros(self.output_shape)

        for filter_idx in range(self.num_filters):
            for input_depth_idx in range(self.filters.shape[1]):
                output[filter_idx] += signal.correlate2d(input[input_depth_idx],
                                                         self.filters[filter_idx, input_depth_idx], mode="valid")

            output[filter_idx] = relu(output[filter_idx] + self.biases[filter_idx])

        return output

    def backward(self, output_grad, learning_rate):

        """
        Thinking: The output of a conv layer is a 3D feature map. This means the accompaning gradients for that
        feature map will be 3D as well. So, output_grad will be a 3D matrix of gradients. Now each 2D feature map within that
        3D structure will have its own filter. This filter is going to have the same depth as the input to this layer.
        """

        filter_gradients = np.zeros(self.filters.shape)
        input_gradients = np.zeros(self.input.shape)

        for filter_idx in range(self.num_filters):
            # doing calculations on a per filter basis
            for input_depth_index in range(self.filters.shape[1]):
                filter_gradients[filter_idx, input_depth_index] = signal.correlate2d(self.input[input_depth_index],
                                                                                     output_grad[filter_idx],
                                                                                     mode="valid")
                input_gradients[input_depth_index] += signal.convolve2d(output_grad[filter_idx],
                                                                        self.filters[filter_idx, input_depth_index],
                                                                        mode="full")

        # we adjust the biases by using the sum(output_gradient[some filter index])
        for filter_idx in range(self.num_filters):
            self.biases[filter_idx] -= learning_rate * np.sum(output_grad[filter_idx])

        # adjust filters using filter_gradients
        self.filters -= learning_rate * filter_gradients

        return input_gradients


class Pooling:
    def __init__(self, input_shape):
        # Input
        self.input_shape = input_shape
        input_depth, input_height, input_width = input_shape

        # Filter
        self.filter_shape = (input_depth, 2, 2)
        self.stride_length = 2

        # Output
        self.output_shape = (input_depth, input_height // 2, input_width // 2)

        # Store indices of picked max values
        self.indices = {}

    def forward(self, input):
        output = np.zeros(self.output_shape)

        """
        filter (fixed to 2x2):
        a b
        c d
        """

        # It works, but probably not the efficient way of doing it.
        for depth in range(len(input)):
            height = 0

            a = [0, 0]
            b = [0, 1]
            c = [1, 0]
            d = [1, 1]

            outputRow = 0
            outputCol = 0

            while d[0] < len(input[0]) and d[1] < len(input[0][0]):
                maxNum = max(input[depth, a[0], a[1]], input[depth, b[0], b[1]], input[depth, c[0], c[1]],
                             input[depth, d[0], d[1]])

                maxOptionsIndices = [(depth, a[0], a[1]), (depth, b[0], b[1]), (depth, c[0], c[1]), (depth, d[0], d[1])]
                maxOptions = [input[depth, a[0], a[1]], input[depth, b[0], b[1]], input[depth, c[0], c[1]],
                              input[depth, d[0], d[1]]]
                maxNumIdx = np.argmax(np.array(maxOptions))

                output[depth, outputRow, outputCol] = maxNum

                self.indices[(depth, outputRow, outputCol)] = maxOptionsIndices[maxNumIdx]

                outputCol += 1

                # move to the left
                a[1] += self.stride_length
                b[1] += self.stride_length
                c[1] += self.stride_length
                d[1] += self.stride_length

                if d[1] >= len(input[0][0]):
                    height += 2
                    outputRow += 1
                    outputCol = 0

                    a = [height, 0]
                    b = [height, 1]
                    c = [height + 1, 0]
                    d = [height + 1, 1]

        return output

    def backward(self, output_grad):
        """
        Basic idea: We will be given the max-pooled output in the form of a gradient.
        """

        """
        input_max_indices structure:
        2D
        each row is a different layer
        """

        # output_grad will have the same shape as self.output_shape
        input_grad = np.zeros(self.input_shape)

        for depth in range(len(output_grad)):
            for row in range(len(output_grad[0])):
                for col in range(len(output_grad[0][0])):
                    input_grad[self.indices[(depth, row, col)]] = output_grad[depth, row, col]

        return input_grad


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        # Need to flatten to column vector.
        # flatten returns row vector, then reshape to column vector
        # reshape param: -1 means a row for each element, 1 means one column
        return np.ndarray.flatten(input).reshape(-1, 1)

    def backward(self, output_grad, prev_weights):
        base_partial = np.dot(prev_weights.transpose(), output_grad)

        return base_partial.reshape(self.input_shape)


class Dense:
    def __init__(self, num_neurons, input_shape, activation):
        self.num_neurons = num_neurons

        self.weights = np.random.randn(self.num_neurons, input_shape[0]) * np.sqrt(2.0 / input_shape[0])
        self.biases = np.zeros((num_neurons, 1))

        if activation == "relu":
            self.activation_function = relu
        elif activation == "softmax":
            self.activation_function = softmax

    def forward(self, input):
        self.input = input
        self.z = np.dot(self.weights, input) + self.biases
        self.output = self.activation_function(self.z)

        return self.output

    def backward(self, base_partial_from_right_layer, weights_from_right_layer, learning_rate, desired_output=None):
        if self.activation_function == softmax:
            basePartial = cost_delta_output_layer(self.output, desired_output)

            weightGrad = np.dot(basePartial, self.input.transpose())
            biasesGrad = basePartial

            self.weights -= learning_rate * weightGrad
            self.biases -= learning_rate * biasesGrad

            return basePartial
        elif self.activation_function == relu:
            # BASE: bias gradients of previous neurons * the previous weights * sigmoid_deriv of current neuron (with current z)
            prevWeights = weights_from_right_layer

            reluDeriv = relu_derivative(self.z)

            basePartial = np.dot(prevWeights.transpose(), base_partial_from_right_layer) * reluDeriv

            # weight: BASE * activation of next (LEFT) layer
            weightGrad = np.dot(basePartial, self.input.transpose())

            # bias: BASE
            biasesGrad = basePartial

            self.weights -= learning_rate * weightGrad
            self.biases -= learning_rate * biasesGrad

            return basePartial

        # base partial is what gets propagated between dense layers


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z < 0, 0, 1)


def softmax(output):
    # subtract the largest element to reduce possible overflow error, does not alter probability distribution
    exp_values = np.exp(output - np.max(output))

    return exp_values / np.sum(exp_values)

def categorical_cross_entropy_cost(output, desired):
    # Avoid numerical instability by clipping the output values to a small positive value
    output = np.clip(output, 1e-10, 1 - 1e-10)

    # Calculate the categorical cross-entropy cost
    cost = -np.sum(desired * np.log(output))

    return cost


def cost_delta_output_layer(output, desired):
    # Chain rule for softmax & cross entropy simplifies to this nice form - ref: https://www.youtube.com/watch?v=rf4WF-5y8uY
    return output - desired
