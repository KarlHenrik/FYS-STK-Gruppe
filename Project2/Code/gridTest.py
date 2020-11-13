import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score

def gridTest(etas, lmdas, createModel, params, metrics, x_train, x_test, y_train, y_test, usingTensorFlow=False):
    # Setting up correct metrics function and arrays to hold results
    if metrics == "accuracy" or metrics == "accuracy_skl":
        metrics_func = accuracy_score
    elif metrics == "mse":
        metrics_func = mean_squared_error

    train_mse = np.zeros((len(etas), len(lmdas)))
    test_mse = np.zeros((len(etas), len(lmdas)))
    # Creating and testing a model for each combination of lmda and eta
    for i, eta in enumerate(etas):
        for j, lmda in enumerate(lmdas):
            model = createModel(eta, lmda, params)
            if usingTensorFlow:
                model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=0)
            else:
                model.fit(x_train, y_train)
            y_train_mdl = model.predict(x_train)
            y_mdl = model.predict(x_test)

            if metrics == "accuracy": # Turning classification results to onehot vectors
                y_train_mdl = (y_train_mdl > 0.5).astype(int)
                y_mdl = (y_mdl > 0.5).astype(int)
            train_mse[i][j] = metrics_func(y_train_mdl, y_train)
            test_mse[i][j] = metrics_func(y_mdl, y_test)
    # Plotting results
    sns.set(font_scale=1.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    sns.heatmap(train_mse, annot=True, ax=ax1, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy" or metrics == "accuracy_skl":
        ax1.set_title("Training Accuracy")
    elif metrics == "mse":
        ax1.set_title("Training MSE")
    ax1.set_ylabel("$\eta$")
    ax1.set_xlabel("$\lambda$")

    sns.heatmap(test_mse, annot=True, ax=ax2, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy" or metrics == "accuracy_skl":
        ax2.set_title("Test Accuracy")
    elif metrics == "mse":
        ax2.set_title("Test MSE")
    ax2.set_ylabel("$\eta$")
    ax2.set_xlabel("$\lambda$")
    ax2.set_ylabel("$\eta$")
    ax2.set_xlabel("$\lambda$")
    plt.show()
    
def gridCV(etas, lmdas, createModel, params, metrics, X, y, kfolds = 5, usingTensorFlow=False):
    # Setting up correct metrics function and arrays to hold results
    if metrics == "accuracy" or metrics == "accuracy_skl":
        metrics_func = accuracy_score
    elif metrics == "mse":
        metrics_func = mean_squared_error

    train_mse = np.zeros((len(etas), len(lmdas)))
    test_mse = np.zeros((len(etas), len(lmdas)))
    # Creating and testing a model for each combination of lmda and eta
    for i, eta in enumerate(etas):
        for j, lmda in enumerate(lmdas):
            for train_inds, test_inds in _split(X, kfolds):
                x_train = X[train_inds]
                x_test = X[test_inds]
                #x_train, X_test = scale(X_train, X_test)

                y_train = y[train_inds]
                y_test = y[test_inds]
                
                model = createModel(eta, lmda, params)
                if usingTensorFlow:
                    model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=0)
                else:
                    model.fit(x_train, y_train)
                y_train_mdl = model.predict(x_train)
                y_mdl = model.predict(x_test)

                if metrics == "accuracy": # Turning classification results to onehot vectors
                    y_train_mdl = (y_train_mdl > 0.5).astype(int)
                    y_mdl = (y_mdl > 0.5).astype(int)
                train_mse[i][j] += metrics_func(y_train_mdl, y_train) / kfolds
                test_mse[i][j] += metrics_func(y_mdl, y_test) / kfolds
    # Plotting results
    sns.set(font_scale=1.8)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy" or metrics == "accuracy_skl":
        ax.set_title("Training Accuracy")
    elif metrics == "mse":
        ax.set_title("Training MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy" or metrics == "accuracy_skl":
        ax.set_title("Test Accuracy")
    elif metrics == "mse":
        ax.set_title("Test MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
    
def mseCV(model, X, y, kfolds = 5):
    mse = 0
    for train_inds, test_inds in _split(X, kfolds):
        x_train = X[train_inds]
        x_test = X[test_inds]

        y_train = y[train_inds]
        y_test = y[test_inds]

        model.fit(x_train, y_train)
        y_mdl = model.predict(x_test)
        
        mse += mean_squared_error(y_mdl, y_test) / kfolds
    return mse
    
def _split(data, k):
    n = len(data)
    fold_size = n // k #standard fold size, will be one larger if we need to get rid of extra elements
    test_start = 0
    extra = n % k #the first extra folds need one more element
    
    fold_indexes = []
    for i in range(k):
        if extra > 0:
            test_size = fold_size + 1
            extra -= 1
        else:
            test_size = fold_size
        training_size = n - test_size
        test_stop = test_start + test_size
        
        training_indexes = np.zeros(training_size, dtype=int)
        training_indexes[:test_start] = np.array(range(0, test_start)) #before testing
        training_indexes[test_start:] = np.array(range(test_stop, n)) #after testing
 
        testing_indexes = np.array(range(test_start, test_stop))

        fold_indexes.append([training_indexes, testing_indexes])
        test_start += test_size
        
    return fold_indexes