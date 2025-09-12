'''
Computational Physics HW 1
'''

import numpy as np

'''
Part 1) Differentiation
    a) Differentiate functions cos(x) and exp(x) at x = 0.1, 10 using single precision
    forward-, central-, and extrapolated-difference algorithms.
    b) Make Error Plot
    c) Identify and discuss the different regimes in which
    truncation and roundoff error manifests in these plots
'''

# Define functions
def cos(x):
    return np.cos(x, dtype=np.float32)
def exp(x):
    return np.exp(x, dtype=np.float32)

def forward_diff(f, x):
    h = np.float32(1e-4)
    '''
    h is step size, it comes from the equation h=(4C|f(x)/f''(x)|)^0.5
    where C is machine precision at around 10^-7 for single precision.
    Since |f(x)/f''(x)| for both cos(x) and exp(x) is 1, plugging these
    values in to the equation gives h ~ 10^-4
    '''
    return (f(x+h) - f(x)) / h

def central_diff(f, x):
    h = np.float32(1e-2)
    return (f(x+h/2)-f(x-h/2)) / h

# Define variables
x_vals = [0.1, 10]


forward_results_cos = []
forward_results_exp = []

for x in x_vals:
    forward_results_cos.append(forward_diff(cos,x))
    forward_results_exp.append(forward_diff(exp,x))

central_results_cos = []
central_results_exp = []

for x in x_vals:
    central_results_cos.append(central_diff(cos,x))
    central_results_exp.append(central_diff(exp,x))

# Compute actual results
act_results_cos = []
act_results_exp = []

for x in x_vals:
    act_results_cos.append(-np.sin(x, dtype=np.float32))
    act_results_exp.append(np.exp(x, dtype=np.float32))

print(forward_results_cos)
print(forward_results_exp)
print(central_results_cos)
print(central_results_exp)
print(act_results_cos)
print(act_results_exp)
