import numpy as np
from pylab import plot, show, xlabel, ylabel, title, legend

a = 1.0
b = 3.0
x0 = 0.0
y0 = 0.0
delta = 1e-10
t0 = 0.0
t1 = 20.0
H = 20.0
nmax = 8

r = np.array([x0,y0], float)
tpoints = [t0]
xpoints = [r[0]]
ypoints = [r[1]]

def f(r):
    x = r[0]
    y = r[1]
    dxdt = 1 - (b+1) * x + a * x**2 * y
    dydt= b*x -a*x**2*y
    return np.array([dxdt,dydt],dtype=float)



def Buliersh_Stoer_recursion(r, t, H):
    # Recursive method to implement the adaptice BS method
    # Used afte initialization of first row in BS method

    def modified_midpoint(r, n):
        r2 = np.copy(r)
        h = H/n
        r1 = r2 + 0.5 * h * f(r2)
        r2 += h * f(r1)
        for i in range(n-1):
            r1 += h * f(r2)
            r2 += h * f(r1)
        return 0.5 * (r2 + r1 + 0.5 * h * f(r2))


    def compute_row(R1, n):
        if n > 8:
            # divide in half and try again
            r1 = Buliersh_Stoer_recursion(r, t, H/2)
            return Buliersh_Stoer_recursion(r1, t + H / 2, H/2)
        else:
            # Compute first value
            R2 = [modified_midpoint(r, n)]
            # Compute rest of row
            for m in range(2, n+1):
                R2.append(R2[m-2]+(R2[m-2]-R1[m-2])/((n/(n-1))**(2*(m-1))-1))
            
            # compute error
            R2 = np.array(R2, float)
            temp = (R2[n-2]-R1[n-2])/((n/(n-1)) ** (2*(n-1))-1)
            error = np.sqrt(temp[0]** 2 + temp[1]**2)

            target = H * delta
            if error < target:
                tpoints.append(t+H)
                xpoints.append(R2[n-1][0])
                ypoints.append(R2[n-1][1])
                return R2[n-1]
            else:
                return compute_row(R2, n+1)
    return compute_row(np.array([modified_midpoint(r,1 )], float), 2)

Buliersh_Stoer_recursion(r, t0, H)


plot(tpoints, xpoints, label="x(t)")
plot(tpoints, ypoints, label="y(t)")

# put the dots right on the curves (interpolating via last accepted values)
# since we stored every accepted end in tpoints/xpoints/ypoints already,
# we can just overplot circular markers there:
plot(tpoints, xpoints, "o", label="interval end (x)")
plot(tpoints, ypoints, "o", label="interval end (y)")

xlabel("t")
ylabel("Concentration")
title("Brusselator: adaptive Bulirsch–Stoer")
legend()
show()
