#The class GDOptimizers contains functionality for batch SGD and batch momentum SGD. It hopefully works for mimimizing functions with parameters
#theta in any shape. If you want to expand the functionality to include AdaGrad or RMSProp which works for a theta with any shape you will need
#to use a very different structure from what was used for SGD and momentumSGD. If the parameters theta are all in a 1D vector however it will be easy.
import numpy as np

class GDOptimizers:
    def __init__(self, learning_rate, epochs, batch_size, optimizer = "SGD", alpha = 0):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        if optimizer == "SGD":
            self.optimizer = self.SGD
        if optimizer == "momentum":
            self.optimizer = self.momentumSGD
            self._alpha = alpha

    def SGD(self, X, z, theta, gradFunc, params = {}):
        n_inputs = len(z)
        # del opp i minibatches
        data_indices = np.arange(n_inputs)
        iterations = n_inputs // self._batch_size
        for i in range(self._epochs):
            for j in range(iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(data_indices, size = self._batch_size, replace=False)
                # minibatch training data
                Xbatch = X[chosen_datapoints]
                zbatch = z[chosen_datapoints]
                # finding gradient
                g = gradFunc(Xbatch, zbatch, theta)
                theta = theta - self._learning_rate * g
        return theta
    
    def momentumSGD(self, X, z, theta, gradFunc, params = {}):
        n_inputs = len(z)
        # del opp i minibatches
        data_indices = np.arange(n_inputs)
        iterations = n_inputs // self._batch_size
        v = theta * 0 # to get same shape as theta
        for i in range(self._epochs):
            for j in range(iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(data_indices, size = self._batch_size, replace=False)
                # minibatch training data
                Xbatch = X[chosen_datapoints]
                zbatch = z[chosen_datapoints]
                # finding gradient
                g = gradFunc(Xbatch, zbatch, theta, params)
                v = self._alpha * v + self._learning_rate * g
                theta = theta - v
        return theta