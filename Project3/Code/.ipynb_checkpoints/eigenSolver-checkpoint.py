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
    def __init__(self, A, x0, dt, T, eta = 0.01):
        self._A = A
        N = x0.shape[0]
        x0 /= np.linalg.norm(x0) # normalize
        self._x0 = tf.reshape(tf.convert_to_tensor(x0, dtype=tf.dtypes.float64), shape=(1, -1)) # row vector, since the NN outputs row vectors
        
        Nt = int(T / dt) + 1
        self._t_arr = np.linspace(0, T, Nt)
        self._t = tf.reshape(tf.convert_to_tensor(self._t_arr, dtype=tf.dtypes.float64), shape=(-1, 1)) # column vector
        
        self._zeros = tf.convert_to_tensor(np.zeros((N, Nt)))

        # Setting up model
        self._model = Sequential()
        self._model.add(Dense(100, activation='sigmoid'))
        self._model.add(Dense(50, activation='sigmoid'))
        self._model.add(Dense(25, activation='sigmoid'))
        self._model.add(Dense(N, activation="linear"))
        self._model.build(self._t.shape)

        self._optimizer = optimizers.Adam(eta)
    
    def _x_net(self):
        return tf.exp(-self._t) @ self._x0 + self._model(self._t) * (1 - tf.exp(-self._t))
    
    def _loss(self):
        with tf.GradientTape() as tape_t:
            tape_t.watch([self._t])
            x_net = self._x_net()
        dx_dt = tape_t.batch_jacobian(x_net, self._t)[:, :, 0] # This takes the gradient of each element of x for each time step

        dx_dt = tf.transpose(dx_dt) # We need to transpose, as x_net is a collection of row vectors,
        x_net = tf.transpose(x_net) # but we need a collection of column vectors for the matrix multiplications

        Ax = self._A @ x_net
        xTx = tf.einsum("ij,ji->i", tf.transpose(x_net), x_net)
        xTAx = tf.einsum("ij,ji->i", tf.transpose(x_net), Ax)
        fx = xTx * Ax + (1 - xTAx) * x_net

        return tf.losses.mean_squared_error(self._zeros, dx_dt - fx + x_net)
        
    def train(self, iters):
        # Training the model by calculating gradient
        for i in range(iters):
            with tf.GradientTape() as tape:
                current_loss = self._loss()

            grads = tape.gradient(current_loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
            
        # Output of model
        return self._t_arr, self._x_net()
        
    def output(self):
        # Output of model
        return self._t_arr, self._x_net()