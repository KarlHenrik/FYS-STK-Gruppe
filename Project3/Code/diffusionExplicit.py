import numpy as np

def expScheme(dx, dt, L, time):
    Nx = int(L / dx) + 1
    x = np.linspace(0, L, Nx)

    Nt = int(time / dt) + 1
    t = np.linspace(0, time, Nt)

    alpha = dt/dx**2

    U = np.zeros((Nx, Nt)) # Holds values at all time steps
    
    U[:, 0] = np.sin(np.pi * x) # Initial conditions

    for j in range(Nt - 1): # tid
        for i in range(1, Nx - 1): # sted
            U[i, j+1] = alpha * U[i-1, j] + (1 - 2 * alpha) * U[i, j] + alpha * U[i+1, j]

    return x, t, U

        