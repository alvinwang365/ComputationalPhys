import numpy as np
import matplotlib.pyplot as plt

M = 100 # number of cells per side
L = 100.0 # box length
dx = L/M # cell size
q = -1.0 # charge

# load positions
particles = np.loadtxt("particles.dat")
x = particles[:,0]
y = particles[:,1]

# setting up charge density/cloud in cell technique
rho = np.zeros((M,M), dtype = float)

for xp, yp in zip(x, y):
    # adjust positions ot be relative to center of cells
    # (positions are not always integer values)
    sx = xp - 0.5
    sy = yp - 0.5

    # prepare to distribute charge to center of cells
    # initialize from bottom left cell
    i = int(np.floor(sx)) # finds the floor integer in x -> left
    j = int(np.floor(sy)) # finds the floor integer in y -> bottom

    # Need offset to distribute charge accurately, 
    # find fraction of charge that should be left behind as a result of this offset
    # Note fraction 0<f<1
    fx = sx - i
    fy = sy - j

    # distribute charge using initialized information, should be distributed to four surrounding cell centers
    for di in (0,1):
        ii = i + di #selects cell center
        if ii < 0 or ii>= M:
            continue #discarding charge at boundaries since boundary is grounded
        # calculate weight of charge to place in cell center. we decided to leave fx behind 
        wx = (1.0 - fx) if di == 0 else fx

        for dj in (0,1):
            jj = j + dj
            if jj < 0 or jj >= M:
                continue
            wy = (1.0-fy) if dj==0 else fy

            weight = wx*wy

            rho[jj, ii] += q * weight

plt.figure(figsize=(6, 5))
plt.imshow(rho, origin="lower",
           extent=[0, L, 0, L],
           interpolation="nearest")
plt.colorbar(label="Charge density (arb. units)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Charge density field (cloud-in-cell)")
plt.tight_layout()
plt.show()

# PART B
#Choosing units so that epsilon0 is also 1
epsilon0 = 1
def solve_poisson_relaxation(rho, M, dx, eps0=1.0, tol =1e-10, max_iter = 1000000):
    phi = np.zeros_like(rho) #initial guess of 0 everywhere satisfies boundary conditions
    for it in range(max_iter):
        phi_new = phi.copy()

        #update only interior points
        for j in range(1, M-1):
            for i in range(1, M-1):
                phi_new[j, i] = 0.25 * (phi[j, i+1] + phi[j, i-1]+
                                        phi[j+1, i] + phi[j-1,i]+
                                        dx*dx * rho[j,i]/eps0)
                
        # check differenve for with maximum change btn iterations
        diff = np.max(np.abs(phi_new-phi))
        phi = phi_new
        
        if diff< tol:
            return phi, it + 1
        
    #if not converged by max iter
    return phi, max_iter


# Solve relaxation method
phi_relax, n_iter_relax = solve_poisson_relaxation(rho, M, dx, eps0=epsilon0)
# print iterations
print(f"Part (b): standard relaxation converged in {n_iter_relax} iterations.")

# graph field
plt.figure(figsize=(5, 4))
plt.imshow(phi_relax, origin="lower",
           extent=[0, L, 0, L],
           interpolation="nearest")
plt.colorbar(label="Potential φ (arb. units)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Part (b): Potential field from standard relaxation")
plt.tight_layout()
plt.show()

# PART C

def solve_poisson_GS(rho, M, dx, omega, eps0 =1.0, tol = 1e-10, max_iter = 1000000):
    phi = np.zeros_like(rho)

    for it in range(max_iter):
        max_diff = 0.0
        for j in range (1, M-1):
            for i in range (1,M-1):
                phi_GS = 0.25 * (phi[j,i+1]+phi[j,i-1]+ phi[j+1,i]+phi[j-1,i]+dx*dx*rho[j, i]/eps0)

                # Update with over relaxation
                phi_new = (1.0-omega)*phi[j,i]+omega*phi_GS

                diff = abs(phi_new - phi[j,i])
                if diff > max_diff:
                    max_diff = diff

                phi[j,i] = phi_new
        
        if max_diff < tol:
            return phi, it + 1
    
    return phi, max_iter

# helper function to find iterations
def iterations(omega):
    _, iters = solve_poisson_GS(rho, M, dx, omega, eps0=epsilon0)
    return iters

# Golden ratio search  initialization for optimized omega (minimized number of iterations for convergence)
phi_opt = None
iters_opt = None

omega_low = 1.0
omega_high = 2.0
tol_omega = 0.001

gr = (np.sqrt(5.0)-1.0)/2.0 # golden ratio

# initial interior points
c = omega_high - gr * (omega_high - omega_low)
d = omega_low + gr*(omega_high - omega_low)

f_c = iterations(c)
f_d = iterations(d)

omega_history = []
best_omega_history = []

# Goldnen ratio search
while (omega_high - omega_low) > tol_omega:
    # record current midpoint as our best estimate so far
    omega_mid = 0.5 * (omega_low + omega_high)
    omega_history.append(omega_mid)

    if f_c < f_d:
        omega_high = d
        d, f_d = c, f_c
        c = omega_high - gr * (omega_high - omega_low)
        f_c = iterations(c)
    else:
        omega_low = c
        c, f_c = d, f_d
        d = omega_low + gr * (omega_high - omega_low)
        f_d = iterations(d)

    # store the current best ω (the midpoint)
    best_omega_history.append(0.5 * (omega_low + omega_high))

omega_opt = 0.5 * (omega_low + omega_high)

# final solve with optimized parameters
phi_GS_opt, iters_opt = solve_poisson_GS(rho, M, dx, omega_opt, eps0=epsilon0)

print(f"Part (c): optimal ω ≈ {omega_opt:.3f}, "
      f"converged in {iters_opt} iterations.")

# plot omega during minimization process
plt.figure(figsize=(5, 4))
plt.plot(best_omega_history, marker="o")
plt.xlabel("Golden-ratio iteration step")
plt.ylabel("Best ω estimate")
plt.title("Part (c): Evolution of ω during golden ratio search")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot potential using GS
plt.figure(figsize=(5, 4))
plt.imshow(phi_GS_opt, origin="lower",
           extent=[0, L, 0, L],
           interpolation="nearest")
plt.colorbar(label="Potential φ (arb. units)")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Part (c): Potential field with optimized Gauss-Seidel Method (ω ≈ {omega_opt:.3f})")
plt.tight_layout()
plt.show()