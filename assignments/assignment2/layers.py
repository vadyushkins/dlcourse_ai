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
