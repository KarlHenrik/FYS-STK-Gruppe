#The FFNN lets you create a feed forward neural network with a given number of layers with a given number of nodes and given activation funcitons
import numpy as np

class FFNN:
    def __init__(self, inputs):
        self._inputs = inputs
        
        self._theta = np.array([], dtype=object)
        self._activations = []
        self._diffs = []
        self._layers = 0
        
    def addLayer(self, neurons, activation):
        if (self._layers == 0):
            inputs = self._inputs
        else:
            inputs = len(self._theta[self._layers - 1][1][0]) # number of nodes in previous layer
            
        weights = np.random.randn(inputs, neurons) / inputs
        bias = np.zeros((1, neurons)) + 0.01
        
        theta = self._theta.tolist()
        theta.append([weights, bias])
        self._theta = np.array(theta, dtype=object) # theta is an array filled with nested arrays of different sizes
        
        if activation == "sigmoid":
            self._activations.append(self._sigmoid)
            self._diffs.append(self._sigmoid_diff)
        elif activation == "linear":
            self._activations.append(self._linear)
            self._diffs.append(self._one_diff)
        elif activation == "relu":
            self._activations.append(self._relu)
            self._diffs.append(self._relu_diff)
        elif activation == "leakyrelu":
            self._activations.append(self._leakyrelu)
            self._diffs.append(self._leakyrelu_diff)
        elif activation == "softmax":
            self._activations.append(self._softmax)
            self._diffs.append(self._one_diff)
        else:
            raise ValueError("Unsupported activation function.") # ADD SPECIFICS
            
        self._layers += 1
        
    def compile(self, optimizer, loss = "mse", lmda = 0):
        # setting up parameters for the fitting
        self._gd = optimizer
        self._lmda = lmda
        self._g0 = self._theta * 0
        # Checking for unsupported structure of network
        if loss not in {"mse", "cross"}:
            raise ValueError("Unsupported cost function. Try \"mse\" or \"cross\"")
        for layer in range(self._layers - 1):
            if self._activations[layer] == self._softmax:
                raise ValueError("Softmax in hidden layer not supported")
        if loss == "mse" and self._activations[-1] == self._softmax:
            raise ValueError("Softmax not supported for loss = \"mse\", try \"cross\"")
        if loss == "cross" and self._activations[-1] not in {self._softmax, self._sigmoid}:
            raise ValueError("Only Softmax and sigmoid are supported for output layer for loss = \"cross\"")
        if loss == "cross" and self._activations[-1] == self._sigmoid and len(self._theta[self._layers - 1][1]) != 2:
            raise ValueError("Sigmoid output layer with loss =\"cross\" is only supported for two outputs")
        # Changing sigmoid derivative if it is output and loss = "cross"
        if loss == "cross" and self._activations[-1] == self._sigmoid:
            self._diffs[-1] = self._one_diff
            
        
    def _feedForward(self, a_in, theta):
        self._as = [a_in] # _as[i] comes into layer i. is the output of layer i - 1
        self._zs = [] # _zs[i] comes into layer i activation
        for i in range(self._layers):
            weights = theta[i][0]
            bias = theta[i][1]
            z = self._as[-1] @ weights + bias
            self._zs.append(z)
            self._as.append(self._activations[i](z))
        
    def _backProp(self, a_in, target, theta):
        # setting up all outputs in the network
        self._feedForward(a_in, theta)
        # setting up gradient "array"
        g = self._g0
        n = target.shape[0]
        # output layer
        delta = (self._as[-1] - target) * self._diffs[-1](self._zs[-1], self._as[-1]) / n # cost' * activation'
        g[self._layers - 1][0] = self._as[-2].T @ delta  # weight gradient
        g[self._layers - 1][1] = np.sum(delta, axis = 0) # bias gradient
        if self._lmda > 0: # regularization
            g[self._layers - 1][0] += theta[self._layers - 1][0] * 2 * self._lmda
        # the rest of the layers
        for i in range(self._layers - 2, -1, -1): #from the second to last, to the first layer
            #theta[i+1][0] is layer i+1 weights. self._zs[i] is the input to layer i activation. self._as[i + 1] is the output of layer i activation
            delta = delta @ theta[i+1][0].T * self._diffs[i](self._zs[i], self._as[i + 1])
            g[i][0] = self._as[i].T @ delta # weight gradient. self._as[i] is the input to layer i
            g[i][1] = np.sum(delta, axis = 0) # bias gradient
            if self._lmda > 0: # regularization
                g[i][0] += theta[i][0] * 2 * self._lmda
        return g
    
    def fit(self, a_in, target):
        self._theta = self._gd.optimizer(a_in, target, self._theta, self._backProp)
        return
    
    def predict(self, a_in):
        self._feedForward(a_in, self._theta)
        return self._as[-1]
            
    # ---------- Activation functions and their derivative functions --------------
    def _sigmoid(self, z):
        #return 1 / (1 + np.exp(-z))
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), 
                                np.exp(z) / (1 + np.exp(z)))
        
    def _sigmoid_diff(self, z, a): # z is usual input, a is usual output
        return a * (1 - a)
    
    def _linear(self, z):
        return z
        
    def _one_diff(self, z, a): # z is usual input, a is usual output
        return 1
    
    def _softmax(self, z):
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def _relu(self, z):
        return np.where(z > 0, z, 
                               0)
        
    def _relu_diff(self, z, a): # z is usual input, a is usual output
        return np.where(z > 0, 1, 
                               0)
    
    def _leakyrelu(self, z):
        return np.where(z > 0, z, 
                               0.01 * z)
        
    def _leakyrelu_diff(self, z, a): # z is usual input, a is usual output
        return np.where(z > 0, 1, 
                               0.01)