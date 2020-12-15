# FYS-STK4155 Project 3

## Code Folder
For this project we have written code implementing TensorFlow neural networks and training them with tf.GradientTape and the Adam optimizer to solve partial differential equations(and a non-linear differential equation).

### diffusionExplicit.py
Implements the explicit sheme. See Diffusion.ipynb for use case and syntax.

### diffusionUtils.py
Contains utility functions to plot solutions to the diffusion equation, compute analytical solution, and compare to analytical solution. See Diffusion.ipynb for use case and syntax.

### diffusionTf.py
Class which lets you create a neural network model specialized to solving the diffusion equation using automatic differentiation. Must be intialized and then trained with a number of iterations. See Diffusion.ipynb for use case and syntax.

### eigenSolver.py
Class which lets you create a neural network model specialized to solving the eigenvalue problem using automatic differentiation. Must be intialized and then trained with a number of iterations. See Eigenvalue.ipynb for use case and syntax.

## Report Folder

Contains our report pdf.