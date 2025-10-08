'''
Computational HW 2 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    return (x[0]-2)**2 + (x[1]-2)**2

def back_grad(f, vec, h = 1e-5):
    grad = np.zeros_like(vec)
    for i in range(len(vec)):
        vec_back = vec.copy()
        vec_back[i] -= h
        grad[i] = (f(vec) - f(vec_back)) / h
    return grad

def grad_desc(f, init_vec, lr = 0.1, tol = 1e-6, max_iter = 1000):
    vec = init_vec
    values = [f(vec)]
    for i in range(max_iter):
        grad = back_grad(f, vec)
        next_vec = vec - lr * grad
        values.append(f(next_vec))
        if np.linalg.norm(next_vec - vec) < tol:
            break
        vec = next_vec
    return vec, values
    
# Run gradient descent
start_point = np.array([5.0, -3.0])
min_point, history = grad_desc(f, start_point)

# Print result
print("Minimum found at:", min_point)
print("Function value at minimum:", f(min_point))

# Plot convergence
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("f(x, y)")
plt.title("Convergence of Gradient Descent using Numerical Derivatives")
plt.grid(True)
plt.show()

# Schechter function part

# Load data
df = pd.read_csv("smf_cosmos.dat", delim_whitespace=True, header=None)
df.columns = ['logMgal', 'nMgal', 'error']

logMgals = df['logMgal'].values
nMgals = df['nMgal'].values
errors = df['error'].values

def Schechter (logMgal, freeparams):
    logphistar, logMstar, alpha = freeparams
    return np.power(10,logphistar)*np.power(np.power(10,logMgal)/np.power(10,logMstar),alpha+1)*np.exp(-np.power(10,logMgal)/np.power(10,logMstar))*np.log(10) # note the logs here are base 10

    return sum
def chi2func(nMgals, errors, logMgals):
    def func(freeparams):
        sum = 0
        for i in range(len(nMgals)):
            numerator = ( nMgals[i] - Schechter(logMgals[i], freeparams))**2
            denominator = errors[i]**2
            sum+= (numerator/denominator)
        return sum
    return func

# need to rewrite gradient descent functions to fit the way our chi2 is written
def back_grad_chi(f, nMgals, errors, logMgals, chi, vec, h = 1e-5):
    grad = np.zeros_like(vec)
    for i in range(len(vec)):
        vec_back = vec.copy()
        vec_back[i] -= h
        grad[i] = (chi(nMgals, errors, logMgals,f, vec) - chi(nMgals, errors, logMgals, f, vec_back)) / h
    return grad

def grad_desc_chi(f,chi, init_vec, nMgals, errors, logMgals, lr = 0.1, tol = 1e-6, max_iter = 1000):
    vec = init_vec
    values = [chi(nMgals, errors, logMgals,f,  vec)]
    for i in range(max_iter):
        grad = back_grad_chi(f, nMgals, errors, logMgals, chi, vec)
        next_vec = vec - lr * grad
        values.append(chi(nMgals, errors, logMgals, f, next_vec))
        if np.linalg.norm(next_vec - vec) < tol:
            break
        vec = next_vec
    return vec, values

init_params = np.array([-3.2, 11.5, -0.5])

Xfunc = chi2func(nMgals, errors, logMgals)
min_point_chi, history_chi = grad_desc(Xfunc, init_params, lr = 1e-4, tol = 0.001)

initial_model = Schechter(logMgals,init_params)
optimized_model = Schechter(logMgals,min_point_chi)

print("For initial params [log phi, log Mgals, alpha]:", init_params, "optimized params are:", min_point_chi)

# test other starting points
init_params2 = np.array([-3.2, 10.5, -0.4])
min_point_chi2, history_chi2 = grad_desc(Xfunc, init_params2, lr = 1e-4, tol = 0.001)
print("For initial params [log phi, log Mgals, alpha]:", init_params2, "optimized params are:", min_point_chi2)

init_params3 = np.array([-2.5, 11.0, -0.5])
min_point_chi3, history_chi3 = grad_desc(Xfunc, init_params3, lr = 1e-4, tol = 0.001)
print("For initial params [log phi, log Mgals, alpha]:", init_params3, "optimized params are:", min_point_chi3)

init_params4 = np.array([-3.0, 10.0, -0.67])
min_point_chi4, history_chi4 = grad_desc(Xfunc, init_params4, lr = 1e-4, tol = 0.001)
print("For initial params [log phi, log Mgals, alpha]:", init_params4, "optimized params are:", min_point_chi4)

# chi^2 as a function of step i
# Create iteration array
iterations = np.arange(len(history_chi))
iterations2 = np.arange(len(history_chi2))
iterations3 = np.arange(len(history_chi3))
iterations4 = np.arange(len(history_chi4))

# Plot chi^2 vs iteration
plt.figure(figsize=(8, 6))
plt.plot(iterations, history_chi, marker='o', label='Start [log phi, log Mgals, alpha] = [-3.2, 11.5, -0.5]')
plt.plot(iterations2, history_chi2, marker='o', label='Start [log phi, log Mgals, alpha] = [-3.2, 10.5, -0.4]')
plt.plot(iterations3, history_chi3, marker='o', label='Start [log phi, log Mgals, alpha] = [-2.5, 11.0, -0.5]')
plt.plot(iterations4, history_chi4, marker='o', label='Start [log phi, log Mgals, alpha] = [-3.0, 10.0, -0.67]')
plt.xlabel("Iteration (i)")
plt.ylabel(r"$\chi^2$")
plt.title(r"Comparison of $\chi^2$ Minimization for Different Starting Points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# best-fit Schechter compared with given data on log-log plot
plt.figure(figsize=(7, 5))
plt.errorbar(logMgals, np.log10(nMgals), 
             yerr=(errors / (nMgals * np.log(10))), 
             fmt='o', label='COSMOS Data', capsize=3)

plt.plot(logMgals, np.log10(initial_model), label='Initial Guess', linestyle='--', color='gray')
plt.plot(logMgals, np.log10(optimized_model), label='Optimized Fit', color='purple')

plt.xlabel(r"$\log_{10}(M_{\mathrm{gal}})$")
plt.ylabel(r"$\log_{10}(n(M_{\mathrm{gal}}))$")
plt.title("Schechter Function Fit: Initial vs Optimized\n([log phi, log Mgals, alpha] = [-3.2, 11.5, -0.5])")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


