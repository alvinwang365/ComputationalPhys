'''
Computational Physics HW 1
'''

import numpy as np
import matplotlib.pyplot as plt

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

def forward_diff(f, x, h):
    # optimized h = (1e-4)
    return (f(x+h) - f(x)) / h

def central_diff(f, x, h):
    # optimized h = (1e-2)
    return (f(x+h/2)-f(x-h/2)) / h

def extrap_diff(f, x, h):
    # optimized h = (4e-2)
    fprime1 = ( ( f( x + h ) - f( x - h ) ) ) / ( 2 * h ) 
    fprime2 = ( f( x + h*2 ) - f( x - h*2 ) ) / (4 * h)
    return (4 * fprime1 - fprime2) / 3

# Define variables
x_vals = [np.float32(0.1), np.float32(10.0)]

# Create step sizes, use np.logspace function to create array
# of 100 points between 1e-0.5 and 1e-7 which is a good range for 
# testing step sizes
h_vals = np.logspace(-7, -0.5, 100).astype(np.float32)

errors = {
    'cos': {'forward': [], 'central':[], 'extrap': []},
    'exp': {'forward': [], 'central':[], 'extrap': []}
}

for x in x_vals:
    actual_cos = -np.sin(x, dtype=np.float32)
    actual_exp = exp(x)

    front_cos_errs, cent_cos_errs, ex_cos_errs = [], [], []
    front_exp_errs, cent_exp_errs, ex_exp_errs = [], [], []

    for h in h_vals:
        temp_fr_cos = forward_diff(cos,x , h)
        temp_ct_cos = central_diff(cos, x, h)
        temp_ex_cos = extrap_diff(cos, x, h)

        front_cos_errs.append(abs( (temp_fr_cos - actual_cos ) / actual_cos))
        cent_cos_errs.append(abs( (temp_ct_cos - actual_cos ) / actual_cos))
        ex_cos_errs.append(abs( (temp_ex_cos - actual_cos) / actual_cos))

        temp_fr_exp = forward_diff(exp,x , h)
        temp_ct_exp = central_diff(exp, x, h)
        temp_ex_exp = extrap_diff(exp, x, h)

        front_exp_errs.append(abs( (temp_fr_exp - actual_exp) / actual_cos))
        cent_exp_errs.append(abs( (temp_ct_exp - actual_exp) / actual_exp))
        ex_exp_errs.append(abs( (temp_ex_exp - actual_exp) / actual_exp))
    
    errors['cos']['forward'].append(front_cos_errs)
    errors['cos']['central'].append(cent_cos_errs)
    errors['cos']['extrap'].append(ex_cos_errs)

    errors['exp']['forward'].append(front_exp_errs)
    errors['exp']['central'].append(cent_exp_errs)
    errors['exp']['extrap'].append(ex_exp_errs)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12,10))
titles = ['cos(x)', 'exp(x)']
methods = ['forward', 'central', 'extrap']

for i, func in enumerate(['cos', 'exp']):
    for j, x in enumerate(x_vals):
        ax = axes[i][j]
        for method in methods:
            ax.loglog(h_vals, errors[func][method][j], label=f'{method.capitalize()}', marker='.')
        ax.set_title(f'{titles[i]} at x = {x}')
        ax.set_xlabel('Step size h')
        ax.set_ylabel('Relative error ε')
        ax.grid(True, which="both", ls="--")
        ax.legend()

plt.suptitle("Log-Log Plots of Relative Error vs Step Size", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

