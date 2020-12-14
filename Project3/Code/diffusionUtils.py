import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(x, t, U):
    dx = x[1] - x[0]
    
    plt.title(f'dx = {dx}')
    for t_i in [0, int(len(t) * 0.1), int(len(t) * 0.3), int(len(t) * 0.4), int(len(t) * 0.8)]:
        plt.plot(x, U[:, t_i], label = f'Time =  {t[t_i]:0.3f}')
    plt.xlabel('x [m]')
    plt.ylabel('$u$ ')
    plt.legend()
        
def plot3D(x, t, U):
    dx = x[1] - x[0]
    T, X = np.meshgrid(t, x)

    fig = plt.figure()
    ax = Axes3D(fig)
    #fig = plt.figure(projection='3d')
    ax.set_title(f'dx = {dx}')
    ax.plot_surface(T, X, U, linewidth=0, antialiased=False, cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$')
    ax.set_zlabel('u')
    
def analyticSolv(dx, dt, L, time):
    Nx = int(L / dx) + 1
    x = np.linspace(0, L, Nx)

    Nt = int(time / dt)
    t = np.linspace(0, time, Nt)

    U = np.zeros((Nx, Nt)) # Holds values at all time steps

    for j in range(Nt): # time
        U[:, j] = np.sin(np.pi * x) * np.exp(-np.pi**2 * t[j])

    return x, t, U

def error(x, t, U):
    U_analytic = U * 0
    for j in range(len(t)):
        U_analytic[:, j] = np.sin(np.pi * x) * np.exp(-np.pi**2 * t[j])
        
    abs_err = np.abs(U_analytic - U)
    max_abs_err = np.amax(abs_err)
    mean_abs_err = np.mean(abs_err)
    print(f"Max abs error: {max_abs_err}")
    print(f"Mean abs error: {mean_abs_err}")