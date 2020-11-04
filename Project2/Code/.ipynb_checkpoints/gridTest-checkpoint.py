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
                model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0)
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
    sns.set(font_scale=1.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    sns.heatmap(train_mse, annot=True, ax=ax1, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy":
        ax1.set_title("Training Accuracy")
    elif metrics == "mse":
        ax1.set_title("Training MSE")
    ax1.set_ylabel("$\eta$")
    ax1.set_xlabel("$\lambda$")

    sns.heatmap(test_mse, annot=True, ax=ax2, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy":
        ax2.set_title("Test Accuracy")
    elif metrics == "mse":
        ax2.set_title("Test MSE")
    ax2.set_ylabel("$\eta$")
    ax2.set_xlabel("$\lambda$")
    ax2.set_ylabel("$\eta$")
    ax2.set_xlabel("$\lambda$")
    plt.show()
    
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score

def gridTest(etas, lmdas, createModel, params, metrics, x_train, x_test, y_train, y_test, usingTensorFlow=False):
    # Setting up correct metrics function and arrays to hold results
    if metrics == "accuracy":
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
                model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0)
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
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_mse, annot=True, ax=ax, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy":
        ax.set_title("Training Accuracy")
    elif metrics == "mse":
        ax.set_title("Training MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_mse, annot=True, ax=ax, cmap="viridis_r", xticklabels=np.log10(lmdas), yticklabels=np.log10(etas))
    if metrics == "accuracy":
        ax.set_title("Test Accuracy")
    elif metrics == "mse":
        ax.set_title("Test MSE")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
"""