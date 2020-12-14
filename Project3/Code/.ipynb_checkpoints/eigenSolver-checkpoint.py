import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
tf.keras.backend.set_floatx('float64')

class EigenSolver():
    def __init__(self, dx, dt, L, time, eta = 0.01):
        # Setting up data
        self._Nx = int(L / dx) + 1
        self._Nt = int(time / dt) + 1

        self._x_np = np.linspace(0, L, self._Nx)
        self._t_np = np.linspace(0, time, self._Nt)

        X, T = np.meshgrid(self._x_np, self._t_np)

        x = X.ravel()
        t = T.ravel()

        self._zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1,1))
        self._x = tf.reshape(tf.convert_to_tensor(x), shape=(-1,1))
        self._t = tf.reshape(tf.convert_to_tensor(t), shape=(-1,1))

        # Setting up model
        self._model = Sequential()
        self._model.add(Dense(20, activation='sigmoid'))
        self._model.add(Dense(1, activation="linear"))
        self._model.build(tf.concat([self._x, self._t], 1).shape)

        self._optimizer = optimizers.Adam(lr = eta)
    
    def _g_trial(self):
        return (1 - self._t) * tf.sin(np.pi * self._x) + self._x * (1 - self._x) * self._t * self._model(tf.concat([self._x, self._t], 1))
    
    def loss(model, x0, t):
        with tf.GradientTape() as tape_t:
            tape_t.watch([t])
            x_net = tf.exp(-t) @ x0 + model(t) * (1-tf.exp(-t))
        dx_dt = tape_t.batch_jacobian(x_net, t)[:, :, 0] # This takes the gradient of each element of x for each time step

        dx_dt = tf.transpose(dx_dt) # We need to transpose, as x_net is a collection of row vectors,
        x_net = tf.transpose(x_net) # but we need a collection of column vectors for the matrix multiplications

        Ax = A @ x_net
        xTx = tf.einsum("ij,ji->i", tf.transpose(x_net), x_net)
        xTAx = tf.einsum("ij,ji->i", tf.transpose(x_net), Ax)
        fx = xTx * Ax + (1 - xTAx) * x_net

        return tf.losses.mean_squared_error(zeros, dx_dt - fx + x_net)
        
    def train(self, iters):
        # Training the model bu calculating gradient
        for i in range(iters):
            with tf.GradientTape() as tape:
                current_loss = self._loss()

            grads = tape.gradient(current_loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
            
        # Output of model
        return self._x_np, self._t_np, np.array(self._g_trial()).reshape((self._Nt, self._Nx)).T
        
    def output(self):
        # Output of model
        return self._x_np, self._t_np, np.array(self._g_trial()).reshape((self._Nt, self._Nx)).T