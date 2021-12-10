import numpy as np
from numba import njit


EPS = 1e-15


# ReLU
@njit
def ReLU(Y: np.ndarray) -> np.ndarray:
    """ReLU function.
    
    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """

    Z = np.maximum(0, Y)
    return Z

@njit
def ReLU_derivative(Y: np.ndarray) -> np.ndarray:
    """ReLU derivative function.

    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """
    Z = Y.copy()
    y = Z.shape[0]
    x = Z.shape[1]
    for i in range(y):
        for k in range(x):
            if Z[i, k] <= 0:
                Z[i, k] = 0
            else:
                Z[i, k] = 1
    return Z


@njit
def sigmoid(Y: np.ndarray) -> np.ndarray:
    """Sigmoid function.

    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """

    Y = Y.copy()
    Z = 1 / (1 + np.exp(-Y))
    return Z

@njit
def sigmoid_derivative(Y: np.ndarray) -> np.ndarray:
    """Sigmoid derivative function.

    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """

    Z = sigmoid(Y) * ((1) - sigmoid(Y))
    return Z


# TODO
def softmax(Y: np.ndarray) -> np.ndarray:
    """Softmax function.

    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """

    Y = Y.copy()
    Z = 1 / (1 + np.exp(-Y))
    return Z

# TODO
def softmax_derivative(Y: np.ndarray) -> np.ndarray:
    """Softmax derivative function.

    Args:
        Y: Input array.
        
    Returns:
        Z: Result.
    """

    Z = sigmoid(Y) * (1 - sigmoid(Y))
    return Z


# Binary Cross Entropy
@njit
def binary_crossentropy(Z: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    """Binary Cross Entropy function.

    Args:
        Z: True labels array (n_samples, n_features).
        Z_pred: Predicted values array (n_samples, n_features).
        
    Returns:
        L: Result.
    """

    # Set up
    Z_pred = Z_pred.copy()

    # Handle input
    y = Z_pred.shape[0]
    x = Z_pred.shape[1]
    for i in range(y):
        for k in range(x):
            if Z_pred[i, k] == 0:
                Z_pred[i, k] = EPS
            elif Z_pred[i, k] == 1:
                Z_pred[i, k] = 1 - EPS

    # Estimate function value
    L = Z * np.log(Z_pred) + (1 - Z) * np.log(1 - Z_pred)
    return -L

@njit
def binary_crossentropy_derivative(Z: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    """Binary Cross Entropy derivative function.

    Args:
        Z: True labels array (n_samples, n_features).
        Z_pred: Predicted values array (n_samples, n_features).
        
    Returns:
        L: Result.
    """

    # Set up
    Z_pred = Z_pred.copy()

    # Handle input
    y = Z_pred.shape[0]
    x = Z_pred.shape[1]
    for i in range(y):
        for k in range(x):
            if Z_pred[i, k] == 0:
                Z_pred[i, k] = EPS
            elif Z_pred[i, k] == 1:
                Z_pred[i, k] = 1 - EPS

    # Estimate function value
    L = Z / Z_pred - (1 - Z) / (1 - Z_pred)
    return -L


@njit
def multiclass_crossentropy(Z: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    """Cross Entropy function.

    Args:
        Z: True labels array (n_samples, n_features).
        Z_pred: Predicted values array (n_samples, n_features).
        
    Returns:
        L: Result.
    """

    # Set up
    Z_pred = Z_pred.copy()

    # Handle input
    y = Z_pred.shape[0]
    x = Z_pred.shape[1]
    for i in range(y):
        for k in range(x):
            if Z_pred[i, k] == 0:
                Z_pred[i, k] = EPS

    # Estimate function value
    L = Z * np.log(Z_pred)
    return -1 * L

@njit
def multiclass_crossentropy_derivative(Z: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    """Cross Entropy derivative function.

    Args:
        Z: True labels array (n_samples, n_features).
        Z_pred: Predicted values array (n_samples, n_features).
        
    Returns:
        L: Result.
    """

    # Set up
    Z_pred = Z_pred.copy()

    # Handle input
    y = Z_pred.shape[0]
    x = Z_pred.shape[1]
    for i in range(y):
        for k in range(x):
            if Z_pred[i, k] == 0:
                Z_pred[i, k] = EPS

    # Estimate function value
    L = Z / Z_pred
    return -1 * L


# Tools
@njit
def to_class(Z: np.ndarray) -> np.ndarray:
    """Classification function.

    Classificates raw predicted values.

    Args:
        Z: Predicted values array (n_samples, n_features).
        
    Returns:
        V: Result.
    """

    # Copy
    V = Z.copy()

    # Turn
    V[V < 0.5] = 0
    V[V >= 0.5] = 1

    return V

@njit
def estimate_accuracy(V: np.ndarray, V_pred: np.ndarray) -> float:
    """Accuracy function.

    Estimates accurace between labels.

    Args:
        V: True labels array (n_samples, n_features).
        V_pred: Predicted classes array (n_samples, n_features).
        
    Returns:
        Accuracy: Prediction accuracy.
    """

    arg1 = V.argmax(1).reshape(-1, 1)
    arg2 = V_pred.argmax(1).reshape(-1, 1)

    count = 0
    y = arg1.shape[0]
    x = arg1.shape[1]
    for i in range(y):
        for k in range(x):
            if arg1[i, k] == arg2[i, k]:
                count += 1

    return count / arg1.size

@njit
def hot_encoding(V: np.ndarray) -> np.ndarray:
    """One-hot encoding function.

    Codes classes to one-hot vectors.

    Args:
        V: True labels array (n_samples, n_features).
        
    Returns:
        Z: One-hot vectors.
    """

    Z = np.zeros(
        (V.size, 10),
        dtype='float64'
    )

    for i in range(V.size):
        Z[i, V[i, 0]] = 1

    return Z