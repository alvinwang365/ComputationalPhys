'''
Computational Physics HW 2 part 1
'''

import numpy as np

# Normal relaxation method
def relaxation(f):
    n = 0
    x_test = 1
    print("Initial guess:", x_test)
    while np.abs(f(x_test) - x_test) > 1e-6:
        x_test = f(x_test)
        n+=1
        print(x_test)
    print("Number of iterations for relaxation method:", n)
    return

def over_relax(f):
    n = 0
    x_test = 1
    print("Initial guess:", x_test)
    # Set w
    w = 0.5
    while np.abs(f(x_test) - x_test) > 1e-6:
        x_test = (1+w)*f(x_test) - w*x_test
        n+=1
        print(x_test)
    print("Number of iterations for overrelaxation method, w =", w, ":", n)

    return

def f(x):
    # Set c
    c = 2
    return 1 - np.exp(-c*x)

print("Relaxation Method:")
relaxation(f)
print("Overrelaxation Method:")
over_relax(f)