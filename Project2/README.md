# FYS-STK4155 Project 2

## Code Folder
For this project we have written a neural network class `FFNN`, a linear regression model class `LinearReg` and a logistic regression (for classification) model class `SoftmaxReg`. These all have similar interfaces, with `fit()` and `predict()` methods which let us easily evaluate them with the same testing functions from the file `gridTest.py`. They require different types of parameters to be initialized however, so compact ways of initializing many models are implemented in the analysis notebooks as we need them.

We have also written a stochastic gradient descent class `GDOptimizers`, which implements stochastic gradient descent with and without momentum.

The general recipe for initializing one of our models is to first create a `GDOptimizers` object, and pass it into the constructor of whatever model you are initializing, together with relevant parameters. The model can then be fit to data with its `fit()` method, and predict outputs with the `predict()` method.

In the case of the neural network, you must add layers with the `addLayer()` method, with a number of nodes and an activation function. The neural network needs to be initialized with the `compile()` method after all the layers have been added. The `GDOptimizers` object is passed to the `compile()` method in the case of the neural network.

We have "example models" for all of our type of models in the notebooks, they should give a good idea of how to use our code.

The Jupyter notebook "Regression.ipynb" contains the testing of our regression models. The Jupyter notebook "Classification.ipynb" contains the testing of our classification models.

The folder "Figures" is where figures from running the notebook were saved.

## Report Folder

Contains our report pdf.

## How to use the neural network code
The neural network has many parameters and settings you can tweak.

When you initialize the network, you must pass the number of inputs to the network as a parameter.

When you add a layer with `addLayer()`, you must pass the number of nodes, and the name of the activation function as parameters. The possible activation functions are "sigmoid", "linear", "relu", "leakyrelu" and "softmax". Softmax is only supported in the output layer of a classification model.

When you have added the layers you want, you must run the `compile()` method. This method takes a `GDOptimizers` object, the name of the loss function ("mse" for mean squared error, or "cross" for cross entropy) and the regularization parameter as parameters. If you loss function is "cross", your model must have the softmax activation in the output layer, of sigmoid activation with only 1 node.

After you have compiled the network, you can use `fit()` and `predict()` to fit and predict data.

We hope our code is understandable!

## How to run our code without it taking an hour

Our final results were calculated with a large number of epochs, and with 5-fold cross validation. To make our code faster, we recommend running `gridTest()` instead of `gridCV()` in our notebooks, as the former runs without cross validation. The former should be a commented line above the other wherever we use `gridCV()`. You can also save some time by reducing the number of epochs in our sgd. The results in our report may not be from the same runs you see in the notebooks now, or from your own runs. The tensorflow models is very slow, and you cannot change the number of epochs without changing `gridTest.py` (due to tensorflow having the epochs specified in the fit function, which doesen't work nicely with our general interface).