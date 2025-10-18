import numpy as np
import matplotlib.pyplot as plt

# Load Data
data = np.loadtxt('sunspots.txt')
months = data[:, 0]
sunspots = data[:, 1]

# Plot Data
plt.figure(figsize=(10, 5))
plt.plot(months, sunspots, color='orange', linewidth=1)
plt.title('Sunspots as a Function of Time')
plt.xlabel('Month')
plt.ylabel('Sunspot Number')
plt.grid(True)
plt.show()

# By inspection of the graph, the cycle is around 130 months

# Calculate DFT
def dft(y):
    """
    Compute the discrete Fourier transform of a 1D array x.
    Returns an array c of complex Fourier coefficients.
    """
    N = len(y)
    c = np.zeros(N//2+1, dtype=complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k]+= y[n]*np.exp(-2j*np.pi*k*n/N)
    return c

# Step 3: Apply DFT to the sunspot data
c = dft(sunspots)
N = len(c)

# Step 4: Compute the power spectrum c_k = |c[k]|^2
c_k = np.abs(c)**2
k_values = np.arange(N)

# Step 6: Find the dominant nonzero peak
dominant_index = np.argmax(c_k[1:]) + 1  # ignore k=0 (DC)
dominant_freq = dominant_index / (len(sunspots))  # cycles per month (divide coefficient by total N to retrieve frequency)
cycle_length_months = 1 / dominant_freq

print(f"Dominant frequency index k = {dominant_index}")
print(f"Estimated cycle length = {cycle_length_months:.1f} months "
      f"({cycle_length_months / 12:.1f} years)")

# Step 5: Plot the power spectrum
plt.figure(figsize=(10, 5))
plt.plot(k_values, c_k, color='blue')
plt.title('Power Spectrum of Sunspot Data')
plt.xlabel('Frequency Index k')
plt.ylabel('$c_k = |c_k|^2$')
plt.grid(True)
plt.show()
