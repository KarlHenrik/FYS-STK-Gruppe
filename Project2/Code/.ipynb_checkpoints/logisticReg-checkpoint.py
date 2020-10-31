import numpy as np

class SoftmaxReg():
    
    def __init__(self, n_in, n_out, lmda):
        weights = np.random.randn(n_in, n_out)
        bias = np.zeros([n_out,1]) + 0.01
        self._theta = np.array([0,0], dtype=object)
        self._theta[0] = weights
        self._theta[1] = bias
        self._g0 = self._theta * 0              # will be used for the gradient
        self._lmda = lmda
        
    def fit(self, x_train, y_train, sgd):
        self._theta = sgd.optimizer(x_train, y_train, self._theta, self._gradient)  
        return self.predict(x_train)
        
    def predict(self, x):
        y = self._softmax(self._theta[0].T @ x.T + self._theta[1])
        return y
     
    def _gradient(self, x, t, theta):
        g = self._g0
        y = self._softmax(theta[0].T @ x.T + theta[1])
        
        delta = (y - t) / t.shape[0]
        g[1] = np.sum(delta, axis = 0)
        g[0] = x.T @ delta
        return g
    
    def _softmax(self, z):
        exps = np.exp(z)
        return exps / np.sum(exps)
    