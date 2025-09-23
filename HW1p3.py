import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# Reading from csv file
file_path = "lcdm_z0.matter_pk"
df = pd.read_csv(file_path, delim_whitespace = True, comment = '#', header = None)

# Designating Column names from csv file
df.columns = ["k", "P_k", "col3", "col4"]

# Storing k values and P(k) values
k_vals = df["k"].values
P_k_vals = df["P_k"].values

# Creating sorted values so Cubic Spline works
sorted_indices = np.argsort(k_vals)
k_vals_sorted = k_vals[sorted_indices]
P_k_vals_sorted = P_k_vals[sorted_indices]

# Interpolating use cubic spline
P_interp = InterpolatedUnivariateSpline(k_vals_sorted, P_k_vals_sorted,k = 3)

# Graphing Interpolated Function

P_k_interp = P_interp(k_vals)
plt.figure(figsize=(8, 6))
plt.loglog(k_vals, P_k_interp, label=r"$P(k)$", color='purple')
plt.xlabel(r"$k\ (h/\mathrm{Mpc})$")
plt.ylabel(r"Interpolated $P(k)\ ((\mathrm{Mpc}/h)^3)$")
plt.title("Interpolated Power Spectrum $P(k)$ vs Wavenumber $k$ (log-log scale)")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Calculting correlation function

def simpsons_rule(f, r, N, start, end):
    h = np.float32((end-start)/N)
    x = np.linspace(start, end, N+1, dtype = np.float32)
    fx = f(x,r)
    # return extended simpsons rule 
    return h/3 * (fx[0] + fx[N] + 4*np.sum(fx[1:N:2]) + 2*np.sum(fx[2:N:2]))

def xi_integrand(k,r):
     return k**2 * P_interp(k) * np.sin(k*r)/(k*r)

def xi_r(r):
     # starting ar an epsilon of 1e-5 to avoid divide by zero
     integral = simpsons_rule(xi_integrand, r, 300000,  1e-5, 100)

     return(1/ (2*np.pi**2)) * integral

r_vals = np.linspace(50, 120, 300)

xi_vals = np.array([xi_r(r) for r in r_vals])
r2_xi_vals = r_vals**2 * xi_vals

plt.figure(figsize = (10,6))
plt.plot(r_vals, r2_xi_vals, label=r"$r^\xi(r)$")
plt.xlabel(r"$r$ (Mpc/h)")
plt.ylabel((r"$r^2 \xi(r)$"))
plt.title("Correlation Function and Observable BAO Peak")
plt.legend()
plt.grid(True)
plt.tight_layout
plt.show()

print("Integral from 1e-5 to 100 for r = 105 with N = 300000:")
integral1 = simpsons_rule(xi_integrand, 105 , 300000, 1e-5, 100)
print(integral1)
print("Integral from 1e-5 to 100 for r = 105 with N = 400000:")
integral2 = simpsons_rule(xi_integrand, 105 , 400000, 1e-5, 100)
print(integral2)
percent = ((integral2-integral1)/integral1)*100
print(f"From N = 300000 to N = 400000 the integral only changes bt {percent} percent.")
print("Integral from 1e-5 to 500 for r = 105 with N = 300000:")
integral3 = simpsons_rule(xi_integrand, 105 , 300000, 1e-5, 500)
print(integral3)
percent2 = ((integral3-integral1)/integral1)*100
print(f"From N = 300000 to N = 400000 the integral only changes bt {percent2} percent.")
