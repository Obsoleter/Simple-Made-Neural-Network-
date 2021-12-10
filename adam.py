from typing import List
import numpy as np

from network import Layer


# Adam Optimizer
class AdamLayer:
    """Class for storing Layers info."""

    def __init__(self) -> None:
        self.M = 0.0
        self.V = 0.0
        self.MB = 0.0
        self.VB = 0.0

class Adam:
    """Adam optimizer."""

    EPS = 1e-8

    def __init__(self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> None:
        """Adam constructor.
        
        Args:
            alpha: Initial learning rate.
            beta1: Decay parameter 1.
            beta2: Decay parameter 2.
        """

        # Parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

        # Iteration
        self.t = 1

    def setup(self, layers: List['Layer']):
        # Set up parameters for layers
        self.layers = []
        for _ in layers:
            self.layers.append(AdamLayer())

    def optimize(self, layers: List['Layer']):
        # Optimize Weights and Bias
        for layer, layer_param in zip(layers, self.layers):
            # Weights
            layer_param.M = self.beta1 * layer_param.M + (1 - self.beta1) * layer.G
            layer_param.V = self.beta2 * layer_param.V + (1 - self.beta2) * (layer.G ** 2)

            # Bias
            layer_param.MB = self.beta1 * layer_param.MB + (1 - self.beta1) * layer.GB
            layer_param.VB = self.beta2 * layer_param.VB + (1 - self.beta2) * (layer.GB ** 2)

            # Alpha
            alpha = self.alpha * np.sqrt(1 - (self.beta2 ** self.t)) / (1 - (self.beta1 ** self.t))
            layer.W -= alpha * layer_param.M / (np.sqrt(layer_param.V) + self.EPS)
            layer.B -= alpha * layer_param.MB / (np.sqrt(layer_param.VB) + self.EPS)

        # Increase iteration number
        self.t += 1