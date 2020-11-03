import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score

def gridTest(etas, lmdas, createModel, params, metrics, x_train, x_test, y_train, y_test):
    if metrics == "accuracy":
        metrics_func = accuracy_score
    elif metrics == "mse":
        metrics_func = mean_squared_error
    
    sns.set(font_scale=1.4)

    train_mse = np.zeros((len(etas), len(lmdas)))
    test_mse = np.zeros((len(etas), len(lmdas)))

    for i, eta in enumerate(etas):
        for j, lmda in enumerate(lmdas):
            model = createModel(eta, lmda, params)

            model.fit(x_train, y_train)
            y_train_mdl = model.predict(x_train)
            y_mdl = model.predict(x_test)

            if metrics == "accuracy":
                y_train_mdl = (y_train_mdl > 0.5).astype(int)
                y_mdl = (y_mdl > 0.5).astype(int)
            train_mse[i][j] = metrics_func(y_train_mdl, y_train)
            test_mse[i][j] = metrics_func(y_mdl, y_test)


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