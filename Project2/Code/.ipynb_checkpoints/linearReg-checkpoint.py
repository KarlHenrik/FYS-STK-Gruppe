#The class LinearReg contains a design matrix and can be used for training a model with parameters beta using a gradient descent method with a regularization parameter (like ridge)
import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearReg():
    def __init__(self, order, lmda):
        self._lmda = lmda
        self._order = order
        self._p = int((order + 1) * (order + 2) / 2)
        self._beta = np.random.randn(self._p, 1)
        self._scaler = StandardScaler() #subtracts mean from each feature and divides by the standard deviation
        self._isFit = False
    
    def designMatrix(self, x, y):
        n = x.size
        X = np.zeros((n, self._p))
        feature = 0
        for x_power in range(self._order + 1):
            for y_power in range(self._order - x_power + 1):
                X[:, feature] = x**x_power * y**y_power
                feature += 1
        return X
    
    def gradient(self, X, z, beta, params):
        return -2 * X.T @ (z - X @ beta) / X.shape[0] + params["lmda"] * 2 * beta
    
    def fit(self, xy, z, sgd):
        x = xy[:, 0]
        y = xy[:, 1]
        X = self.designMatrix(x, y)
        
        self._scaler.fit(X)
        X = self._scaler.transform(X)
        X[:, 0] = 1 # scaling removed the intercept terms
        
        params = {"lmda": self._lmda}
        self._beta = sgd.optimizer(X, z, self._beta, self.gradient, params)
        self._isFit = True
        return X @ self._beta
    
    def predict(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        X_test = self.designMatrix(x, y)
        if self._isFit:
            X_test = self._scaler.transform(X_test)
            X_test[:, 0] = 1 # scaling removed the intercept terms
        return X_test @ self._beta
    
    def ridgeFit(self, xy, z): #in case we want to see what the "analytical" solution gives us
        x = xy[:, 0]
        y = xy[:, 1]
        X = self.designMatrix(x, y)
        
        self._scaler.fit(X)
        X = self._scaler.transform(X)
        X[:, 0] = 1 # scaling removed the intercept terms
        
        self._beta = np.linalg.inv(X.T @ X + self._lmda * np.eye(self._p)) @ X.T @ z
        self._isFit = True
        return X @ self._beta