import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

class diffusionSolver():
    
    def __init__(self, dx, dt, L, time):
        self._dx = dx
        self._dt = dt
        self._Nx = int(L / dx)
        self._Nt = int(time / dt)     
        self._alpha = dt/dx**2
        self.x = np.linspace(0, L, self._Nx)
        self.t = np.linspace(0, time, self._Nt)
        self._U = np.zeros((self._Nx, self._Nt)) # Holds values at all time steps

    def expScheme(self):
        U = self._U
        U[:, 0] = np.sin(np.pi * self.x) # Initial conditions

        for j in range(self._Nt-1): # tid
            for i in range(1, self._Nx-1): # sted
                U[i, j+1] = self._alpha*U[i-1, j] + (1 - 2 * self._alpha) * U[i, j] + self._alpha * U[i+1, j]
        return self.x, self.t, U

    def analyticSolv(self):
        U = self._U
        U[:, 0] = np.sin(np.pi * self.x) # Initial conditions

        for j in range(self._Nt-1): # tid
            for i in range(1, self._Nx-1): # sted
                U[i,j+1] = np.exp(-np.pi**2*self.t[j+1])*np.sin(np.pi*self.x[i])
        return self.x, self.t, U
    
    def plot2D(self, u):
        self._u = u
        t = self.t
        plt.title(f'dx = {self._dx}')
        for t_i in [0, int(len(t) * 0.1), int(len(t) * 0.3), int(len(t) * 0.4), int(len(t) * 0.8)]:
            plt.plot(self.x, self._u[:, t_i], label = f'Time =  {t[t_i]:0.3f}')
        plt.xlabel('x [m]')
        plt.ylabel('$u$ ')
        plt.legend()
        
    def plot3D(self, u):
        self._u = u
        
        T, X = np.meshgrid(self.t, self.x)
        
        fig = plt.figure(projection='3d')
        fig.set_title(f'dx = {self._dx}')
        fig.plot_surface(T, X, self._u, linewidth=0, antialiased=False, cmap=cm.viridis)
        fig.set_xlabel('Time $t$')
        fig.set_ylabel('Position $x$')
        fig.set_zlabel('u')
        
    def quiverplot(self, u):
        self._u = u
        
        T, X = np.meshgrid(self.t, self.x)
        plt.title(f'dx = {self._dx}')
        plt.contourf(X,T, self._u)
        plt.xlabel('Position $x$')
        plt.ylabel('Time $t$')
        