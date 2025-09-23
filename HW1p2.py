'''
Computational Physics HW 1
'''

import numpy as np
import matplotlib.pyplot as plt

'''
Part 2) Integration
'''

# Define exact solution
exact_int = np.float32(1 - np.exp(-1))

# Define functions
def f(t):
    return np.exp(-t).astype(np.float32)

# Following functions will take in N (number of bins) and gives a result that 
# divides integral into N parts and uses the given method to sum up the N parts
def midpoint_rule(N):
    h = np.float32(1.0/N)
    x = np.linspace(h/2, 1 - h/2, N, dtype = np.float32)
    # adding up each h*x value is the same as adding up each x value and multiplying by h
    return h * np.sum(f(x))

def trapezoid_rule(N):
    # create variables and linspace to return result of extended trapezoidal rule (pg. 142)
    h = np.float32(1.0/N)
    x = np.linspace(0, 1, N+1, dtype=np.float32)
    return h * (0.5 * f(x[0]) + 0.5 * f(x[-1]) + np.sum(f(x[1:-1])))

def simpsons_rule(N):
    # First ensure N is even
    if N % 2 ==1:
        N += 1
    h = np.float32(1.0/N)
    x = np.linspace(0, 1, N+1, dtype=np.float32)
    fx = f(x)
    # return extended simpsons rule given on pg 146
    return h/3 * (fx[0] + fx[N] + 4*np.sum(fx[1:N:2]) + 2*np.sum(fx[2:N:2]))

# Creating range of N
N_vals = np.array([2**i for i in range(1, 21)])

mid_err, trap_err, simp_err = [], [], []

for N in N_vals:
    mid_int = midpoint_rule(N)
    trap_int = trapezoid_rule(N)
    simp_int = simpsons_rule(N)
    mid_err.append(abs(mid_int - exact_int) / exact_int)
    trap_err.append(abs(trap_int - exact_int) / exact_int)
    simp_err.append(abs(simp_int - exact_int) / exact_int)

# Plot
plt.figure(figsize=(10,6))
plt.loglog(N_vals, mid_err, label='Midpoint', marker='o')
plt.loglog(N_vals, trap_err, label='Trapezoid', marker='s')
plt.loglog(N_vals, simp_err, label='Simpson', marker='^')
plt.xlabel("Number of bins (N)")
plt.ylabel("Relative error (ε)")
plt.title("Relative error of numerical integration methods")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

mask = (N_vals >= 1) & (N_vals <= 1e1)
log_N = np.log10(N_vals[mask])
log_err_mid = np.log10(mid_err)[mask]
log_err_trap = np.log10(trap_err)[mask]
log_err_simp = np.log10(simp_err)[mask]

# Fit straight lines
slope_mid, intercept_mid = np.polyfit(log_N, log_err_mid, 1)
slope_trap, intercept_trap = np.polyfit(log_N, log_err_trap, 1)
slope_simp, intercept_simp = np.polyfit(log_N, log_err_simp, 1)

# Print slopes
print(f"Midpoint slope ≈ {slope_mid:.2f}")
print(f"Trapezoid slope ≈ {slope_trap:.2f}")
print(f"Simpson slope ≈ {slope_simp:.2f}")