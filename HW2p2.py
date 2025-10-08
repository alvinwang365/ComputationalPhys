'''
Computational Physics HW 2 p2
'''

import numpy as np

# Define Wien displacement constant equation first
def f(x):
    return 5*np.exp(-x)+x -5

# Define Binary Search Root Finding function
def binsearch(x1, x2, f):
    if f(x1)==0:
        return x1
    if f(x2)==0:
        return x2
    while np.abs(x1-x2) > 1e-6:
        mid = 0.5*(x1+x2)
        if (f(mid)<0 and f(x1)) < 0 or (f(mid)>0 and f(x1)>0):
            x1 = mid
        else:
            x2 = mid
    return x1

sol = binsearch(1,5,f)
print(sol, f(sol))

# Calculate surface temperature of sun
h = 6.62e-34
c = 3e8
k_b = 1.38e-23

l = 5.02e-7
b = h*c/(k_b*sol)

T = b/l

print(T)
