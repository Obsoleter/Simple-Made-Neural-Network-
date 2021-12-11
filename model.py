import matplotlib.pyplot as plt
import math_functions as Math

from keras.datasets.mnist import load_data
from sklearn.metrics import ConfusionMatrixDisplay

from network import Network, Layer
from adam import Adam


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
network = Network(784, [
    Layer(500, 'relu'),
    Layer(300, 'relu'),
    Layer(300, 'relu'),
    Layer(10, 'softmax'),
    ], optimizer=Adam(), loss_function='bce', 
    epochs=5, sample_ratio=0.01)

network.fit(X_train, Math.hot_encoding(y_train), True)


# Confusion Matrix and evaluation
# Predict
y_pred = network.predict(X_test, classify=True)

# Test set accuracy
count = y_pred[y_pred == y_test].size
accuracy = count / y_pred.size
print(f"Accuracy: {accuracy}")

# Display matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Testing Data Set\nAccuracy: {accuracy * 100}%", fontdict={'fontweight': 'bold'})
plt.show()