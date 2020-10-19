def SGDLinReg(X, beta, y, eta, batches = 10):
    # del opp i minibatches
    
    
    # getGradient på hver batch
    g = -2 * X.T @ (y - X @ beta)
    
    
    # oppdater beta etter gjennomsnittet av gradientene
    return beta - eta * g