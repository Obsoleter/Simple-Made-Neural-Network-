from typing import Literal, Union

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import resample
from keras.datasets.mnist import load_data


# Math functions
class Math:
    EPS = 1e-15

    # Linear function
    @staticmethod
    def linear(X: np.ndarray, W: np.ndarray, B: np.ndarray):
        Y = np.matmul(X, W) + B
        return Y

    @staticmethod
    def linear_derivative(X: np.ndarray):
        Y = X.copy()
        return Y


    # ReLU
    @staticmethod
    def ReLU(Y: np.ndarray):
        Z = np.maximum(0, Y)
        return Z

    @staticmethod
    def ReLU_derivative(Y: np.ndarray):
        Z = Y.copy()
        Z[Z <= 0] = 0
        Z[Z > 0] = 1
        return Z


    # Sigmoid
    @staticmethod
    def sigmoid(Y: np.ndarray):
        Y = Y.copy()
        Z = 1 / (1 + np.exp(-Y))
        return Z

    @staticmethod
    def sigmoid_derivative(Y: np.ndarray):
        Z = Math.sigmoid(Y) * (1 - Math.sigmoid(Y))
        return Z


    # Binary Cross Entropy
    @staticmethod
    def binary_crossentropy(Z: np.ndarray, Z_pred: np.ndarray):
        # Set up
        Z_pred = Z_pred.copy()

        # Handle input
        Z_pred[Z_pred == 0] = Math.EPS
        Z_pred[Z_pred == 1] = 1 - Math.EPS

        # Estimate function value
        L = Z * np.log(Z_pred) + (1 - Z) * np.log(1 - Z_pred)
        return -L

    @staticmethod
    def binary_crossentropy_derivative(Z: np.ndarray, Z_pred: np.ndarray):
        # Set up
        Z_pred = Z_pred.copy()

        # Handle input
        Z_pred[Z_pred == 0] = Math.EPS
        Z_pred[Z_pred == 1] = 1 - Math.EPS

        # Estimate function value
        L = Z / Z_pred - (1 - Z) / (1 - Z_pred)
        return -L


    # Tools
    @staticmethod
    def to_class(Z: np.ndarray) -> np.ndarray:
        """Returns V"""

        # Copy
        V = Z.copy()

        # Turn
        V[V < 0.5] = 0
        V[V >= 0.5] = 1

        return V

    @staticmethod
    def accuracy(V: np.ndarray, V_pred: np.ndarray) -> float:
        arg1 = V.argmax(1).reshape(-1, 1)
        arg2 = V_pred.argmax(1).reshape(-1, 1)

        count = arg1[arg1 == arg2].size
        return count / arg1.size

    @staticmethod
    def hot_encoding(V: np.ndarray) -> np.ndarray:
        Z = np.zeros(
            (V.size, 10),
            dtype='float64'
        )

        for i in range(V.size):
            Z[i, V[i, 0]] = 1

        return Z


# Neural Network
class Network:
    def __init__(self, input: int, layer: int, output: int, epochs: int = 5, 
            gradient_descent: Union[Literal['stochastic'], Literal['default']] = 'default', sample_ratio: float = 0.01
        ) -> None:
        # Network shape
        self.input = input
        self.layer = layer
        self.output = output

        # Weights
        self.W1 = np.random.uniform(
            size=(input, layer),
            low=1e-3,
            high=0.7e-2,
        )
        self.W2 = np.random.uniform(
            size=(layer, output),
            low=0.2e-2,
            high=1.5e-2
        )

        # Intercepts
        self.B1 = np.full(
            (1, layer),
            0,
            dtype='float64'
        )
        self.B2 = np.full(
            (1, output),
            0,
            dtype='float64'
        )

        # Gradient Descent Params
        self.epochs = epochs
        self.gradient_descent = gradient_descent
        self.sample_ratio = sample_ratio

        self.rateW1 = 0.01
        self.rateW2 = 0.01
        self.rateB1 = 0.1
        self.rateB2 = 0.1

    def fit(self, X_full: np.ndarray, Z_full: np.ndarray, verbose: bool = False):
        acc = None
        N = X_full.shape[0]

        for epoch in range(1, self.epochs + 1):
            # Feed Network
            # Downsample
            if self.gradient_descent == 'stochastic':
                X, Z = resample(X_full, Z_full, n_samples=int(N * self.sample_ratio))
            else:
                X = X_full
                Z = Z_full

            # First layer: ReLU
            Y1 = np.matmul(X, self.W1) + self.B1
            Z1 = Math.ReLU(Y1)

            # Second layer: Sigmoid
            Y2 = np.matmul(Z1, self.W2) + self.B2
            Z2 = Math.sigmoid(Y2)
            Z_pred = Z2.copy()

            # Gradient Descent
            self.gradient(X, Y1, Z1, Y2, Z2, Z)

            # Estimate cost
            Z_pred = self.predict(X)
            entropy = Math.binary_crossentropy(Z, Z_pred)
            cost = entropy.mean()
            acc = Math.accuracy(Z, Z_pred)
            
            # Print info
            if verbose:
                print(f"Epoch: {str(epoch).ljust(10)}Loss: {str(cost).ljust(25)}Accuracy: {str(acc).ljust(25)}")
            
        return acc

    def predict(self, X: np.ndarray, classify: bool = False):
        # First layer: ReLU
        Y1 = np.matmul(X, self.W1) + self.B1
        Z1 = Math.ReLU(Y1)

        # Second layer: Sigmoid
        Y2 = np.matmul(Z1, self.W2) + self.B2
        Z2 = Math.sigmoid(Y2)

        if classify:
            Z2 = Z2.argmax(1).reshape(-1, 1)

        return Z2

    def cost(self, X: np.ndarray, Z: np.ndarray, W1, B1, W2, B2):
        # First layer: ReLU
        Y1 = np.matmul(X, W1) + B1
        Z1 = Math.ReLU(Y1)

        # Second layer: Sigmoid
        Y2 = np.matmul(Z1, W2) + B2
        Z2 = Math.sigmoid(Y2)

        return Math.binary_crossentropy(Z, Z2).mean()

    optimize = True
    def optimize_cv(self, X: np.ndarray, Z: np.ndarray, G1, G1B, G2, G2B):
        if self.optimize:
            self.optimize = False
        else:
            return

        # Params
        L1 = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        L2 = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        L1B = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        L2B = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        # Get costs
        cost = self.cost(X, Z, self.W1, self.B1, self.W2, self.B2)
        best_cost = cost
        best_params = [0.1, 0.1, 0.1, 0.1]

        for l1 in L1:
            for l2 in L2:
                for l1b in L1B:
                    for l2b in L2B:
                        W1 = self.W1 - l1 * G1
                        W2 = self.W2 - l2 * G2
                        B1 = self.B1 - l1b * G1B
                        B2 = self.B2 - l2b * G2B

                        param_cost = self.cost(X, Z, W1, B1, W2, B2)

                        if param_cost < best_cost:
                            best_cost = param_cost
                            best_params = [l1, l2, l1b, l2b]

        self.rateW1 = best_params[0]
        self.rateW2 = best_params[1]
        self.rateB1 = best_params[2]
        self.rateB2 = best_params[3]

    def gradient(self, X: np.ndarray, Y1: np.ndarray, Z1: np.ndarray, Y2: np.ndarray, Z2: np.ndarray, Z: np.ndarray):
        # Output layer
        Ld = Math.binary_crossentropy_derivative(Z, Z2)
        Z2d = Math.sigmoid_derivative(Y2)
        Y2d = Math.linear_derivative(Z1)
        
        D = Ld * Z2d
        G2 = np.matmul( Y2d.T, D )
        G2B = D.mean(0)

        # Hidden layer
        Y2d = Math.linear_derivative(self.W2)
        Z1d = Math.ReLU_derivative(Y1)
        Y1d = Math.linear_derivative(X)
        
        D = Z1d * np.matmul( D, Y2d.T )
        G1 = np.matmul( Y1d.T, D )
        G1B = D.mean(0)

        # Optimizer
        self.optimize_cv(X, Z, G1, G1B, G2, G2B)

        # Batch
        self.W1 -= self.rateW1 * G1
        self.W2 -= self.rateW2 * G2
        self.B1 -= self.rateB1 * G1B
        self.B2 -= self.rateB2 * G2B


# Get MNIST Data
(X_train, y_train), (X_test, y_test) = load_data()

# Reshape
X_train = X_train.reshape(len(X_train), -1).astype('float64')
y_train = y_train.reshape(len(y_train), -1)

X_test = X_test.reshape(len(X_test), -1).astype('float64')
y_test = y_test.reshape(len(y_test), -1)

# Scale
X_train /= 255
X_test /= 255


# NN
network = Network(784, 100, 10, epochs=1000, gradient_descent='stochastic')
network.fit(X_train, Math.hot_encoding(y_train), True)


# Confusion Matrix and evaluation
# Predict
y_pred = network.predict(X_test, True)

# Test set accuracy
count = y_pred[y_pred == y_test].size
accuracy = count / y_pred.size
print(f"Accuracy: {accuracy}")

# Display matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Testing Data Set\nAccuracy: {accuracy * 100}%", fontdict={'fontweight': 'bold'})
plt.show()