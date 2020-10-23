#The FFNN lets you create a feed forward neural network with a given number of layers with a given number of nodes and given activation funcitons
import numpy as np

class FFNN:
    def __init__(self):
        self._layers = []
        
    def addLayer(self, neurons, activation, n_inputs = 0):
        if (n_inputs == 0 or len(self._layers) != 0):
            n_inputs = len(self._layers[-1].bias)
        self._layers.append(self.Layer(n_inputs, neurons, activation))
        
    def feedForward(self, a):
        for layer in self._layers:
            a = layer.feed(a)
            
    
    def backProp(self, target, eta = 0.01):
        o_layer = self._layers[-1]
        err_output = o_layer.a_out - target.reshape(o_layer.a_out.shape)
        o_layer.delta = o_layer.diff() * err_output
        print(o_layer.delta.shape)
        o_layer.weights -= eta * self._layers[-2].a_out.T @ o_layer.delta #nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        o_layer.bias -= eta * o_layer.delta
        
        for i, layer in enumerate(reversed(self._layers[:-1])):
            layer.delta = layer.diff() * layer.weights.T @ self._layers[-(i + 1)].delta
            layer.weights -= eta * layer.delta @ layers_rev[i+2].a_out.T
            layer.bias -= eta * layer.delta
        
    class Layer:
        def __init__(self, n_inputs, neurons, activation):
            self.weights = np.random.randn(n_inputs, neurons)
            self.bias = np.zeros(neurons) + 0.01
            if activation == "sigmoid":
                self._activation = self._sigmoid
                self.diff = self._sigmoid_diff
                
        def feed(self, a_in):
            z = a_in @ self.weights + self.bias
            self.a_out = self._activation(z)
            return self.a_out
        
        def _sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
        def _sigmoid_diff(self):
            return self.a_out * (1 - self.a_out)