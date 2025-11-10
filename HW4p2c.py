import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 1.0
M = 1.0

A = 1.0
B = 1.0

A_values = np.arange(0.5, 10.5, 1) 
B_values = np.arange(0.5, 10.5, 1)

BA = []
t_total = []

for A in A_values:
    for B in B_values:
        BA.append(B/A)
        # Initial velocity = 0.8 velocity of circular orbit
        r0 = 1.0 # initial position is at x = 1.0
        v_circ = math.sqrt(G*M/(4.0*r0))
        v0 = 0.8 * v_circ

        print("A =",A,"B =",B, "B/A =",B/A)

        r = np.array([1.0, 0.0, 0.0, v0],float)
            
        def f(r, t):
            x = r[0]
            y = r[1]
            vx = r[2]
            vy = r[3]
            dx = vx
            dy = vy

            vBH = math.sqrt(vx*vx + vy*vy)

            vdotdfx = (-A/(vBH**3+B))*vx
            vdotdfy = (-A/(vBH**3+B))*vy

            f = -G*M/(x*x + y*y)**(3.0/2.0)
            dvx = f*x + vdotdfx
            dvy = f*y + vdotdfy
            return np.array([dx,dy,dvx,dvy],float)

        a = 0.0
        b = 5.0
        h = 1e-2

        tpoints = np.arange(a,b,h)
        tt = []
        xpoints = []
        ypoints = []

        DELTA = 0.000001

        i = 0
        t = 0

        Ncalc = 0

        while (True):

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

            # break when radius = 10^-7
            if ( (r[0]**2+r[1]**2) < (1e-7)**2):
                t_total.append(t)
                break

BA = np.array(BA)
t_total = np.array(t_total)    


# r vs time
plt.figure(figsize=(7, 4))
plt.plot(BA, t_total, color='crimson', lw=1.0)
plt.xlabel('B/A')
plt.ylabel('Total Time')
plt.title('Total Time as a function of B/A')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
