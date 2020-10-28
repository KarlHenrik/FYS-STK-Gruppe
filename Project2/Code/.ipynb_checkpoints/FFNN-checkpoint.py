#The FFNN lets you create a feed forward neural network with a given number of layers with a given number of nodes and given activation funcitons
import numpy as np

class FFNN:
    def __init__(self, inputs):
        self._inputs = inputs
        self.theta = np.array([], dtype=object)
        self._activations = []
        self._diffs = []
        self._as = [] # _as[i] comes into layer i. is the output of layer i - 1
        self._zs = [] # _zs[i] comes into layer i activation
        self._layers = 0
        self._g0 = np.array([], dtype=object)
        
    def addLayer(self, neurons, activation):
        if (self._layers == 0):
            inputs = self._inputs
        else:
            inputs = len(self.theta[self._layers - 1][1])
            
        weights = np.random.randn(inputs, neurons)
        bias = np.zeros(neurons) + 0.01
        
        theta = self.theta.tolist()
        theta.append([weights, bias])
        self.theta = np.array(theta, dtype=object)
        self._g0 = self.theta * 0
        
        if activation == "sigmoid":
            self._activations.append(self._sigmoid)
            self._diffs.append(self._sigmoid_diff)
            
        self._layers += 1
        
    def _feedForward(self, a_in):
        self._as = [a_in]
        self._zs = []
        for i in range(self._layers):
            weights = self.theta[i][0]
            bias = self.theta[i][1]
            z = self._as[-1] @ weights + bias
            self._zs.append(z)
            self._as.append(self._activations[i](z))
        
    def backProp(self, a_in, target, theta, params = {}):
        # setting up all outputs in the network
        self._feedForward(a_in)
        # setting up gradient "array"
        g = self._g0
        # output layer
        delta = (self._as[-1] - target) * self._diffs[-1](self._zs[-1], self._as[-1]) #(a - t) * sigma(z)'
        g[0][0] = self._as[-2].T @ delta # weight gradient
        g[0][1] = np.mean(delta, axis = 0) # bias gradient
        # the rest of the layers
        for i in range(self._layers - 2, -1, -1): #from the second to last, to the first layer
            #theta[i+1][0] is layer i+1 weights. self._zs[i] is the input to layer i activation. self._as[i + 1] is the output of layer i activation
            delta = delta @ theta[i+1][0].T * self._diffs[i](self._zs[i], self._as[i + 1])
            g[i][0] = self._as[i].T @ delta # weight gradient. self._as[i] is the input to layer i
            g[i][1] = np.mean(delta, axis = 0) # bias gradient
        return g
    
    def fit(self, a_in, target, sgd):
        self.theta = sgd.optimizer(a_in, target, self.theta, self.backProp)
        return self.predict(a_in)
    
    def predict(self, a_in):
        self._feedForward(a_in)
        return self._as[-1]
            
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def _sigmoid_diff(self, z, a): # z is usual input, a is usual output
        return a * (1 - a)