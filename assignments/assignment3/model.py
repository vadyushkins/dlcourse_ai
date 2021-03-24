import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        image_width, image_height, n_channels = input_shape

        filter_size = 3
        padding = 1
        pool_size = 4
        pool_stride = 4

        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, filter_size, padding)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(pool_size, pool_stride)

        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(pool_size, pool_stride)

        left_width = image_width // pool_stride ** 2
        left_height = image_height // pool_stride ** 2

        self.flat = Flattener()
        self.fc = FullyConnectedLayer(left_width * left_height * conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        for k, v in self.params().items():
            v.grad = np.zeros_like(v.grad)

        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.max_pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.max_pool2.forward(out)

        out = self.flat.forward(out)
        out = self.fc.forward(out)

        loss, d_out = softmax_with_cross_entropy(out, y)

        d_out = self.fc.backward(d_out)
        d_out = self.flat.backward(d_out)

        d_out = self.max_pool2.backward(d_out)
        d_out = self.relu2.backward(d_out)
        d_out = self.conv2.backward(d_out)

        d_out = self.max_pool1.backward(d_out)
        d_out = self.relu1.backward(d_out)
        d_out = self.conv1.backward(d_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment

        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.max_pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.max_pool2.forward(out)

        out = self.flat.forward(out)
        out = self.fc.forward(out)

        return np.argmax(out, axis=1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters

        for name, layer in vars(self).items():
            for k, v in layer.params().items():
                result['{}_{}'.format(name, k)] = v

        return result
