import numpy as np

class SoftmaxReg():
    
    def __init__(self):
        
    def designMatrix(self, x, y):
        
    def fit(self, xy, z, sgd):
        
    def predict(self, xy):
        
    def score(self):
     
    def cost(t, h, l=l, X=X, y=y, m=m):
        cost = np.transpose(-y)@np.log(h) - np.transpose(1-y)@np.log(1-h) + (l/2)*np.transpose(t[1:])@t[1:]
        cost = (1/m)*cost
    return cost
        
       