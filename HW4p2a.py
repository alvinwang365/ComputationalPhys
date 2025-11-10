import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 1.0
M = 1.0

# Initial velocity for error tolerance check
r_peri0 = 1e-7
r_apo = 1.0 # initial position is at x = 1.0
semi_major = 0.5*(r_peri0 + r_apo)
eccentricity = (r_apo - r_peri0)/(r_apo + r_peri0)
T = 2.0 * math.pi*math.sqrt(semi_major**3/(G*M/4.0))
v_test = math.sqrt((G*M/4)*(2.0/r_apo - 1.0/semi_major))

print(v_test)

r = np.array([1.0, 0.0, 0.0, v_test],float)

def f(r, t):
    x = r[0]
    y = r[1]
    vx = r[2]
    vy = r[3]
    dx = vx
    dy = vy

    f = -G*M/(x*x + y*y)**(3.0/2.0)
    dvx = f*x
    dvy = f*y
    return np.array([dx,dy,dvx,dvy],float)

a = 0.0
b = 23.0
h = 1e-2

tpoints = np.arange(a,b,h)
tt = []
xpoints = []
ypoints = []

DELTA = 0.000001

i = 0
t = 0

Ncalc = 0
while (t <= b):

    # take first step
    k1 = h * f(r, t)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(r + k3, t + h)
    rt = r + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # take second step
    k1 = h * f(rt, t + h)
    k2 = h * f(rt + 0.5 * k1, t + 0.5 * h + h)
    k3 = h * f(rt + 0.5 * k2, t + 0.5 * h + h)
    k4 = h * f(rt + k3, t + h + h)
    rt1 = rt + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # take step with twice the step size
    k1 = 2 * h * f(r, t)
    k2 = 2 * h * f(r + 0.5 * k1, t + 0.5 * h * 2)
    k3 = 2 * h * f(r + 0.5 * k2, t + 0.5 * h * 2)
    k4 = 2 * h * f(r + k3, t + h * 2)
    rt2 = r + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # calculate rho
    rr1 = np.sqrt(rt1[0]**2 + rt1[1]**2)
    rr2 = np.sqrt(rt2[0]**2 + rt2[1]**2)
    rho = 30 * h * DELTA / np.abs(rr1 - rr2)

    if (rho >= 1):
        i += 1
        t += 2 * h
        r = rt1
        hnew = h * rho**0.25       # adjust new step size
        if (hnew / h > 2):         # prevent h from doubling too fast
            hnew = h * 2
        h = hnew

        xpoints.append(r[0])
        ypoints.append(r[1])
        tt.append(t)

    # if step size too large, reduce h
    if (rho < 1):
        h = h * rho**0.25

    Ncalc += 3

    
xpoints = np.array(xpoints)
ypoints = np.array(ypoints)
tt = np.array(tt)

# Compute r(t)
rpoints = np.sqrt(xpoints**2 + ypoints**2)

# Trajectory plot: y vs x
plt.figure(figsize=(6, 6))
plt.plot(xpoints, ypoints, color='dodgerblue', lw=1.2)
plt.gca().set_aspect('auto')
plt.xlim(-0.1,1.1)
plt.ylim(-0.0005,0.0005)
plt.xlabel('x (arb. units)')
plt.ylabel('y (arb. units)')
plt.title('Orbital trajectory: y vs x (Without Dynamical Friction)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# r vs time
plt.figure(figsize=(7, 4))
plt.plot(tt, rpoints, color='crimson', lw=1.0)
plt.xlabel('time')
plt.ylabel('r = sqrt(x² + y²)')
plt.title('Radial distance vs time (Without Dynamical Friction)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
