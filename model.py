import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
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


# Training
class Network:
    def __init__(self, input: int, layer: int, output: int, epochs: int = 5) -> None:
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

    def fit(self, X: np.ndarray, Z: np.ndarray, verbose: bool = False):
        acc = None

        for epoch in range(1, self.epochs + 1):
            # Feed Network
            # First layer: Sigmoid
            Y1 = np.matmul(X, self.W1) + self.B1
            Z1 = Math.sigmoid(Y1)

            # Second layer: Sigmoid
            Y2 = np.matmul(Z1, self.W2) + self.B2
            Z2 = Math.sigmoid(Y2)
            Z_pred = Z2.copy()


            # Estimate cost
            entropy = Math.binary_crossentropy(Z, Z_pred)
            cost = entropy.mean()
            acc = Math.accuracy(Z, Z_pred)

            # Stochastic Gradient Descent
            self.stochastic_gradient(X, Y1, Z1, Y2, Z2, Z)

            # Print info
            if verbose:
                print(f"Epoch: {str(epoch).ljust(10)}Loss: {str(cost).ljust(25)}Accuracy: {str(acc).ljust(25)}")
            
        return acc

    def predict(self, X: np.ndarray):
        # First layer: Sigmoid
        Y1 = np.matmul(X, self.W1) + self.B1
        Z1 = Math.sigmoid(Y1)

        # Second layer: Sigmoid
        Y2 = np.matmul(Z1, self.W2) + self.B2
        Z2 = Math.sigmoid(Y2)

        return Z2

    def stochastic_gradient(self, X: np.ndarray, Y1: np.ndarray, Z1: np.ndarray, Y2: np.ndarray, Z2: np.ndarray, Z: np.ndarray):
        samples = Z.shape[0]

        for sample in range(samples):
            # Calculate errors for output layer
            Ld = Math.binary_crossentropy_derivative(Z[sample], Z2[sample])
            
            # Get gradient
            Z2d = Math.sigmoid_derivative(Y2[sample])
            Y2d = Math.linear_derivative(Z1[sample])
            G2 = np.matmul( Y2d.reshape(self.layer, 1), (Ld * Z2d).reshape(1, self.output) )
            G2B = (Ld * Z2d).reshape(1, self.output)

            # Calculate errors for hidden layer
            Ld = Ld * Z2d * self.W2
            Ld = Ld.sum(axis=1)
            
            # Get Gradient
            Z1d = Math.sigmoid_derivative(Y1[sample])
            Y1d = Math.linear_derivative(X[sample])
            G1 = np.matmul( Y1d.reshape(self.input, 1), (Ld * Z1d).reshape(1, self.layer) )
            G1B = (Ld * Z1d).reshape(1, self.layer)

            # Steps
            self.W1 -= 0.01 * G1
            self.W2 -= 0.00008 * G2

            self.B1 -= 0.1 * G1B
            self.B2 -= 0.015 * G2B


# Get MNIST Data
(X_train, y_train), (X_test, y_test) = load_data()

X = X_train.reshape(len(X_train), -1).astype('float64')
y = y_train.reshape(len(X_train), -1)

X = X[:1000].astype('float64')
y = y[:1000]

# Scale
X /= 255
y = Math.hot_encoding(y)


# NN
network = Network(784, 100, 10, epochs=200)
network.fit(X, y, True)


# Confusion Matrix and evaluation
# Test data
X = X_test.reshape(len(X_test), -1).astype('float64')
y = y_test.reshape(len(y_test), -1)

X = X[:1000].astype('float64')
y = y[:1000]

# Scale
X /= 255

# Predict
y_pred = network.predict(X)
y_pred = y_pred.argmax(1).reshape(-1, 1)

# Test set accuracy
count = y_pred[y_pred == y].size
accuracy = count / y_pred.size
print(f"Accuracy: {accuracy}")

# Display matrix
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.title(f"Testing Data Set\nAccuracy: {accuracy * 100}%", fontdict={'fontweight': 'bold'})
plt.show()