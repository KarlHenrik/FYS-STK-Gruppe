import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
tf.keras.backend.set_floatx('float64')

class DiffusionTf():
    def __init__(self, dx, dt, L, time, eta = 0.01, extraLayer=False):
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
        if extraLayer:
            self._model.add(Dense(40, activation='sigmoid'))
        self._model.add(Dense(20, activation='sigmoid'))
        self._model.add(Dense(1, activation="linear"))
        self._model.build(tf.concat([self._x, self._t], 1).shape)

        self._optimizer = optimizers.Adam(lr = eta)
    
    def _g_trial(self):
        return (1 - self._t) * tf.sin(np.pi * self._x) + self._x * (1 - self._x) * self._t * self._model(tf.concat([self._x, self._t], 1))
    
    def _loss(self):
        with tf.GradientTape() as tape_x2:
            tape_x2.watch([self._x])
            with tf.GradientTape() as tape_x, tf.GradientTape() as tape_t:
                tape_x.watch([self._x])
                tape_t.watch([self._t])
                g_trial = self._g_trial()

            dg_dx = tape_x.gradient(g_trial, self._x)
            dg_dt = tape_t.gradient(g_trial, self._t)

        dg_d2x = tape_x2.gradient(dg_dx, self._x)

        return tf.losses.mean_squared_error(self._zeros, dg_d2x - dg_dt)
        
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