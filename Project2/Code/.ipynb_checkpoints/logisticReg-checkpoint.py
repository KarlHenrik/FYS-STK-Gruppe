import numpy as np

class SoftmaxReg():
    
    def __init__(self, n_in, n_out, lmda, sgd):
        weights = np.random.randn(n_in, n_out)
        bias = np.zeros([1, n_out]) + 0.01
        self._theta = np.array([0,0], dtype=object)
        self._theta[0] = weights
        self._theta[1] = bias
        self._g0 = self._theta * 0              # will be used for the gradient
        self._lmda = lmda
        self._sgd = sgd
        
    def fit(self, x, y):
        self._theta = self._sgd.optimizer(x, y, self._theta, self._gradient)
        return
        
    def predict(self, x):
        y = self._softmax(x @ self._theta[0] + self._theta[1])
        return y
     
    def _gradient(self, x, t, theta):
        g = self._g0
        y = self._softmax(x @ theta[0] + theta[1])

        delta = (y - t) / t.shape[0]
        g[0] = x.T @ delta + theta[0] * 2 * self._lmda
        g[1] = np.sum(delta, axis = 0)
        return g
    
    def _softmax(self, z):
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    