import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output

    Returns:
      probs, np array of the same shape as predictions - probability for every class, 0..1
    """

    if predictions.ndim == 1:
        normalized_predictions = predictions - np.max(predictions)
        return np.exp(normalized_predictions) / np.sum(np.exp(normalized_predictions))
    else:
        normalized_predictions = predictions - np.max(predictions, axis=1)[:, np.newaxis]
        return np.exp(normalized_predictions) / np.sum(np.exp(normalized_predictions), axis=1)[:, np.newaxis]


def cross_entropy_loss(probabilities, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probabilities, np array, shape is either (N) or (batch_size, N) - probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    if probabilities.ndim == 1:
        return -np.log(probabilities[target_index])
    else:
        flatten_target_index = target_index.flatten()
        return -np.mean(np.log(probabilities[(
            np.arange(flatten_target_index.shape[0]),
            flatten_target_index
        )]))


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) - classifier output
      target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probabilities = softmax(predictions)
    loss = cross_entropy_loss(probabilities, target_index)

    subtr = np.zeros_like(probabilities)

    if probabilities.ndim == 1:
        subtr[target_index] = 1
        dprediction = probabilities - subtr
    else:
        batch_size = predictions.shape[0]
        subtr[(
            np.arange(target_index.shape[0]),
            target_index.flatten()
        )] = 1
        dprediction = (probabilities - subtr) / batch_size

    return loss, dprediction


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * (W ** 2).sum()
    gradient = 2 * reg_strength * W

    return loss, gradient


def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    """
    predictions = np.dot(X, W)

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)

    return loss, dW


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        return np.maximum(self.X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        r = d_out.copy()
        r[self.X <= 0] = 0
        return r

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return self.X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad += self.X.T.dot(d_out)
        self.B.grad += d_out.sum(axis=0, keepdims=0)

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        """
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        """

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = Param(X.copy())

        out_height = height + 1 - self.filter_size + 2 * self.padding
        out_width = width + 1 - self.filter_size + 2 * self.padding

        out = np.zeros([batch_size, out_height, out_width, self.out_channels])

        self.X.value = np.pad(
            array=self.X.value,
            pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant'
        )

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location

                slice_X = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :] \
                    .reshape(batch_size, -1)
                slice_W = self.W.value.reshape(-1, self.out_channels)

                out[:, y, x, :] = slice_X.dot(slice_W) + self.B.value

        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_inp = np.zeros_like(self.X.value)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too

        slice_W = self.W.value.reshape(-1, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                slice_X = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :] \
                    .reshape(batch_size, -1)

                d_inp[:, y:y + self.filter_size, x:x + self.filter_size, :] += \
                    np.dot(d_out[:, y, x, :], slice_W.T) \
                        .reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)

                self.W.grad += \
                    np.dot(slice_X.T, d_out[:, y, x, :]) \
                        .reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)

        return d_inp[:, self.padding:height - self.padding, self.padding:width - self.padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X.copy()

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros([batch_size, out_height, out_width, channels])

        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                out[:, y, x, :] += np.max(
                    X[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :],
                    axis=(1, 2)
                )

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        d_inp = np.zeros_like(self.X)

        batch_idxs = np.repeat(np.arange(batch_size), channels)
        channel_idxs = np.tile(np.arange(channels), batch_size)

        for y in range(out_height):
            for x in range(out_width):
                y_stride = y * self.stride
                x_stride = x * self.stride

                max_idxs = np.argmax(
                    self.X[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :]
                        .reshape(batch_size, -1, channels),
                    axis=1
                )

                slice_d_inp = d_inp[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :] \
                    .reshape(batch_size, -1, channels)

                slice_d_inp[batch_idxs, max_idxs.flatten(), channel_idxs] = d_out[batch_idxs, y, x, channel_idxs]

                d_inp[:, y_stride:y_stride + self.pool_size, x_stride:x_stride + self.pool_size, :] = \
                    slice_d_inp.reshape(batch_size, self.pool_size, self.pool_size, channels)

        return d_inp

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]

        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass

        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
