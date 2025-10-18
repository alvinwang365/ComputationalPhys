import numpy as np
import matplotlib.pyplot as plt

# Load data (assuming one column of closing prices)
data = np.loadtxt('dow.txt')

plt.figure(figsize=(10, 5))
plt.plot(data)
plt.title('Dow Jones Industrial Average (2006–2010)')
plt.xlabel('Trading Days')
plt.ylabel('Closing Value')
plt.grid(True)
plt.show()

# Compute the discrete Fourier transform using rfft
fft_coeffs = np.fft.rfft(data)


ten = int(0.1 * len(fft_coeffs))

freq10 = np.copy(fft_coeffs)
freq10[ten:]=0

inverse10 = np.fft.irfft(freq10, n = len(data))

two = int(0.2 * len(fft_coeffs))

freq2 = np.copy(fft_coeffs)
freq2[two:]=0

inverse2 = np.fft.irfft(freq2, n = len(data) )


# Plot both the original and smoothed signals
plt.figure(figsize=(10, 5))
plt.plot(data, label='Original Data', color='blue', alpha=0.6)
plt.plot(inverse10, label='Low-pass (10%) Reconstructed', color='red', linewidth=2)
plt.title('Dow Jones Industrial Average (2006–2010)\nOriginal vs Low-Pass Reconstructed (10%)')
plt.xlabel('Trading Days')
plt.ylabel('Closing Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(data, label='Original Data', color='blue', alpha=0.6)
plt.plot(inverse10, label='Low-pass (10%) Reconstructed', color='red', linewidth=2)
plt.plot(inverse2, label='Low-pass (2%) Reconstructed', color='lime', linewidth=2)
plt.title('Dow Jones Industrial Average (2006–2010)\nOriginal vs Low-Pass Reconstructed (10 and 2%)')
plt.xlabel('Trading Days')
plt.ylabel('Closing Value')
plt.legend()
plt.grid(True)
plt.show()

# 10 percent cutoff clearly smooths the data with the low pass filter. However, the 2 percent cutoff seems to add oscillations to the curve
# this is a result of the 2 percent filter, filtering out too many frequencies to the point where the constructive and destructive
# intereferences between the remaining sin and cosine functions are unable to cancel each other properly. This shows that higher
# frequency oscillations are still needed to get a smooth function that still accurately represents the shape of the curve.
# the lower remaining frequencies of sin and cosine functions are unable to correct itself in time to recreate an accurate smooth representation
# of the data