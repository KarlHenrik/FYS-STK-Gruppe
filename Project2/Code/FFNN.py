#The FFNN lets you create a feed forward neural network with a given number of layers with a given number of nodes and given activation funcitons
import numpy as np

class FFNN:
    def __init__(self, inputs):
        self._inputs = inputs
        self._theta = np.array([], dtype=object)
        self._activations = []
        self._diffs = []
        self._a = []
        self._layers = 0
        
    def addLayer(self, neurons, activation):
        if (self._layers == 0):
            inputs = self._inputs
        else:
            inputs = len(self._theta[self._layers - 1][1])
            
        weights = np.random.randn(inputs, neurons)
        bias = np.zeros(neurons) + 0.01
        
        theta = self._theta.tolist()
        theta.append([weights, bias])
        self._theta = np.array(theta, dtype=object)
        
        if activation == "sigmoid":
            self._activations.append(self._sigmoid)
            self._diffs.append(self._sigmoid_diff)
            
        self._layers += 1
        
    def feedForward(self, a):
        self._a = [a]
        for i in range(self._layers):
            weights = self._theta[i][0]
            bias = self._theta[i][1]
            #print(f"iteration {i}")
            #print(f"w: {weights}")
            #print(f"b : {bias}")
            #print(f"a: {self._a[-1]}")
            z = self._a[-1] @ weights + bias
            self._a.append(self._activations[i](z))
            
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
            
    def _sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
    def _sigmoid_diff(self):
        return self.a_out * (1 - self.a_out)