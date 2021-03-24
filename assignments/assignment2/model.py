import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)

        self.W_in = self.input_layer.params()['W']
        self.B_in = self.input_layer.params()['B']

        self.W_out = self.output_layer.params()['W']
        self.B_out = self.output_layer.params()['B']

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for p in self.params().values():
            p.grad.fill(0)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        forward_input_layer = self.input_layer.forward(X)
        forward_relu = self.relu.forward(forward_input_layer)
        predictions = self.output_layer.forward(forward_relu)

        loss, dprediction = softmax_with_cross_entropy(predictions, y)

        backward_output_layer = self.output_layer.backward(dprediction)
        backward_relu = self.relu.backward(backward_output_layer)
        backward_input_layer = self.input_layer.backward(backward_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        for p in self.params().values():
            loss_l2, grad_l2 = l2_regularization(p.value, self.reg)
            loss += loss_l2
            p.grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused

        forward_input_layer = self.input_layer.forward(X)
        forward_relu = self.relu.forward(forward_input_layer)
        predictions = self.output_layer.forward(forward_relu)

        return np.argmax(predictions, axis=-1)

    def params(self):
        # TODO Implement aggregating all of the params

        return {
            'W_in': self.W_in,
            'B_in': self.B_in,
            'W_out': self.W_out,
            'B_out': self.B_out,
        }
