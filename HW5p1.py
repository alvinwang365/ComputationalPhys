import numpy as np
import matplotlib.pyplot as plt
from banded import banded

# constants
hbar = 1.0545718e-34
M = 9.109e-31
L = 1.0e-8
N = 1000
a = L/N
h = 1.0e-18

n_steps= 5000

# Crank-Nicolson coefficients
beta = 1j * hbar * h / (2 * M * a**2)
a1 = 1.0 + beta
a2 = -0.5 * beta
b1 = 1 - beta
b2 = 0.5 * beta

# Make initial wave function
x = np.linspace(a, L-a, N-1)

x0 = L/2.0
sigma = 1.0e-10
kappa = 5.0e10

psi = np.exp(-(x-x0)**2 / (2*sigma**2))*np.exp(1j*kappa*x)

# Build A as needed for banded function
n_int = psi.size
up=down=1

A_banded = np.zeros((1+up+down, n_int), dtype = complex)
A_banded[1, :] = a1 # Diagonal
A_banded[0, 1:] = a2 # Upper Diagonal
A_banded[2, :-1] = a2 # Lower Diagonal


#Crank Nicolson step
def crank_nicolson_step(psi, A_band):
    n = psi.size
    v = np.empty_like(psi, dtype = complex)

    # make v vector, the unknowns are actually the psi values at x = a and up to x = L-a, but indices start at 0
    v[0] = b1 * psi[0] + b2 * psi[1] #psi[0-1] in this case is zero
    v[1:-1] = b1 * psi[1:-1]+b2*(psi[2:]+psi[:-2]) # the colons used here slice up the psi vector and distrbutes slices to fill in interior v points
    v[-1] = b1*psi[-1] + b2* psi[-2] #psi[-1 (+1)] is psi[L] which is zero

    psi_new = banded(A_band.copy(), v.copy(), up, down)
    return psi_new

#psi_next = crank_nicolson_step(psi, A_banded) #psi_next contains the next step for the wavefunction at the interior points, can loop this to get the following step

x_plot = np.concatenate(([0.0], x, [L]))
scale = 1.0e-9   # vertical scale factor suggested in the book

y_plot = np.real(np.concatenate(([0.0], psi, [0.0]))) * scale

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x_plot, y_plot)

ax.set_xlabel("x (m)")
ax.set_ylabel("Re(ψ) × {:.0e}".format(scale))
ax.set_xlim(0, L)
ax.set_ylim(1.2 * np.min(y_plot), 1.2 * np.max(y_plot))

#animation loop
for n in range(n_steps):
    psi = crank_nicolson_step(psi, A_banded)

    # update plot data (keep boundaries at zero)
    y_plot[1:-1] = np.real(psi) * scale
    line.set_ydata(y_plot)

    ax.set_title(f"Time step {n+1}, t = {(n+1)*h:.2e} s")
    plt.pause(0.0001)   # controls frame rate

plt.ioff()
plt.show()