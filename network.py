from typing import Any, Literal, Union, List
import numpy as np
import math_functions as Math

from time import time
from sklearn.utils import resample


# Layer
class Layer:
    """Layer class."""

    def __init__(self, size: int, activation_function: Union[Literal['relu'], Literal['sigmoid'], Literal['softmax']]) -> None:
        """Layer constructor.
        
        Args:
            size: Height of layer.
            activation_function: Function of activation.
        """

        # Size
        self.size = size

        # Activation function
        self.activation_function: Any = activation_function
        if self.activation_function == 'relu':
            self.function = Math.ReLU
            self.derivative = Math.ReLU_derivative

        elif self.activation_function == 'sigmoid':
            self.function = Math.sigmoid
            self.derivative = Math.sigmoid_derivative

        elif self.activation_function == 'softmax':
            self.function = Math.softmax
            self.derivative = Math.softmax_derivative

        else:
            raise Exception('Invalid activation function!')

        # Values
        self.Y: Any = None
        self.Z: Any = None
        self.Yd: Any = None
        self.Zd: Any = None
        self.G: Any = None
        self.GB: Any = None

    def setup(self, input: int):
        if self.activation_function == 'sigmoid' or self.activation_function == 'softmax':
            min = -1 * (np.sqrt(6.0) / np.sqrt(input + self.size))
            max = (np.sqrt(6.0) / np.sqrt(input + self.size))
            self.W = np.random.randn(input, self.size)
            self.W = min + self.W * (max - min)

        elif self.activation_function == 'relu':
            std = np.sqrt(2 / input)
            self.W = np.random.randn(input, self.size) * std

        self.B = np.full(
            shape=(1, self.size),
            fill_value=0.0,
            dtype='float64'
        )

    def copy(self):
        # Create
        copy = Layer(self.size, self.activation_function)

        # Copy weights and bias
        copy.W = self.W.copy()
        copy.B = self.B.copy()

        # Gradients
        copy.G = self.G
        copy.GB = self.GB

        return copy

    @staticmethod
    def copy_all(layers: List['Layer']):
        copies = []
        for layer in layers:
            copy = layer.copy()
            copies.append(copy)
        return copies


# Neural Network
class Network:
    """Neural Network class."""

    def __init__(
            self, input: int, layers: List[Layer],
            optimizer, loss_function: Union[Literal['bce'], Literal['mce']],
            epochs: int = 5, sample_ratio: float = 0.01
    ):
        """NN constructor.
        
        Args:
            input: Height of input layer. Usualy number of features.

            layers: List of layers.

            optimizer: Optimizer for learning.

            loss_function: Error function.

            epochs: Number of epochs.

            sample_ratio: Ratio for a Batch size.
        """
        # Init values
        self.input = input
        self.layers = layers
        self.optimizer = optimizer
        
        if loss_function == 'bce':
            self.loss_function = Math.binary_crossentropy
            self.loss_derivative = Math.binary_crossentropy_derivative

        elif loss_function == 'mce':
            self.loss_function = Math.multiclass_crossentropy
            self.loss_derivative = Math.multiclass_crossentropy_derivative

        self.epochs = epochs
        self.sample_ratio = sample_ratio

        # Set up layers
        for layer in self.layers:
            layer.setup(input)
            input = layer.size
        
        # Set up optimizer
        self.optimizer.setup(self.layers)

    def fit(self, X_full: np.ndarray, Z_full: np.ndarray, verbose: bool = False):
        """Fit NN.
        
        Args:
            X_full: Array of samples (n_samples, n_features).
            Z_full: One-hot encoded array of true labels (n_samples, n_classes).
            verbose: Prints out the learning process info.
        """

        # Parameters
        accuracy = None
        N = X_full.shape[0]
        batch_size = int(N * self.sample_ratio)
        batches = N // batch_size

        # Epochs
        for self.epoch in range(1, self.epochs + 1):
            # Epoch parameters
            timer = time()
            cost = 0.0
            accuracy = 0.0

            # Batches, update weights
            for batch in range(batches):
                # Get Batch
                X: Any; Z: Any
                X, Z = resample(X_full, Z_full, n_samples=batch_size)

                # Gradient Descent
                self.gradient(X, Z, self.layers)

                # Optimizer
                self.optimizer.optimize(self.layers)

                # Estimate parameters for epoch, using weights before
                Z_pred = self.predict(X, self.layers)
                cost += self.loss_function(Z, Z_pred).sum(1).mean()
                accuracy += Math.estimate_accuracy(Z, Z_pred)

            # Current epoch values
            timer = time() - timer
            cost /= batches
            accuracy /= batches
                
            # Print epoch info
            if verbose:
                print(f"Epoch: {str(self.epoch).ljust(10)}Loss: {str(cost).ljust(25)}Accuracy: {str(accuracy).ljust(25)}Time: {(str(timer)+'s').ljust(25)}Step: {(str(timer/batches)+'s').ljust(25)}")
            
        return accuracy

    def predict(self, X: np.ndarray, layers: List[Layer] = None, classify: bool = False):
        """Predict.
        
        Args:
            X: Samples array.
            classify: Classify predicted values.
            
        Returns:
            Predicted values. Values are classified if 'classify' is True.
        """

        # Layers
        if layers is None:
            layers = self.layers

        # Feed
        input = X
        for layer in layers:
            Y = np.matmul(input, layer.W) + layer.B
            Z = layer.function(Y)
            input = Z

        # Output
        output = input

        # Classify
        if classify:
            output = output.argmax(1).reshape(-1, 1)

        return output

    def cost(self, X: np.ndarray, Z_true: np.ndarray, layers: List[Layer] = None):
        """Cost function.
        
        Args:
            X: Samples array (n_samples, n_features).
            Z_true: One-hot encoded True labels array (n_samples, n_classes).
        """

        # Layers
        if layers is None:
            layers = self.layers

        # Feed
        input = X
        for layer in layers:
            Y = np.matmul(input, layer.W) + layer.B
            Z = layer.function(Y)
            input = Z

        # Output
        Z_pred = input

        return self.loss_function(Z_true, Z_pred).sum(1).mean()
    
    def gradient(self, X: np.ndarray, Z_true: np.ndarray, layers: List[Layer]):
        # Layers
        if layers is None:
            layers = self.layers

        # Samples
        N = X.shape[0]

        # Forward pass
        input: Any = X
        for layer in layers:
            # Values
            layer.Y = np.matmul(input, layer.W) + layer.B
            layer.Z = layer.function(layer.Y)

            # Derivatives
            layer.Yd = input
            layer.Zd = layer.derivative(layer.Y)

            input = layer.Z
        Z_pred = input
        

        # Backward pass
        Ld = self.loss_derivative(Z_true, Z_pred)
        for layer in reversed(layers):
            # Gradients
            D = layer.Zd * Ld
            layer.G = np.matmul( layer.Yd.T, D ) / N
            layer.GB = D.mean(0)

            # Prev layer errors
            Ld = np.matmul( D, layer.W.T )