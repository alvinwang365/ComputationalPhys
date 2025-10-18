import numpy as np
import matplotlib.pyplot as plt

# Read the grid of brightness values from the file
blur_data = np.loadtxt("blur.txt")

# Display as a density (grayscale) plot
plt.figure(figsize=(8, 8))
plt.imshow(blur_data, cmap='gray', origin='upper')
plt.title("Blurred Image (Gaussian σ = 25)")
plt.colorbar(label='Brightness')
plt.show()

ny, nx = blur_data.shape

# Define sigma
sigma = 25.0

# Create coordinate grids (centered at 0)
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

# Shift coordinates so that (0,0) is at the image center
Xc = X - nx // 2
Yc = Y - ny // 2


# Gaussian
psf = np.exp(-(Xc**2 + Yc**2) / (2 * sigma**2))

# Make it periodic (wraparound in both directions)
psf = (
    psf
    + np.roll(np.roll(psf, nx // 2, axis=1), ny // 2, axis=0)
    + np.roll(psf, nx // 2, axis=1)
    + np.roll(psf, ny // 2, axis=0)
)
psf /= np.sum(psf)  # normalize total intensity to 1


# Display the periodic Gaussian (PSF)
plt.figure(figsize=(8, 8))
plt.imshow(psf, cmap='gray', origin='upper')
plt.title("Periodic Gaussian Point Spread Function (σ = 25)")
plt.colorbar(label='Amplitude')
plt.show()


F_blur = np.fft.rfft2(blur_data)
F_psf = np.fft.rfft2(psf)

epsilon = 1e-3  # threshold to avoid dividing by small numbers

# Create a mask where PSF Fourier coefficients are too small
mask = np.abs(F_psf) < epsilon

# Perform safe division
F_deconv = np.zeros_like(F_blur, dtype=np.complex128)
F_deconv[~mask] = F_blur[~mask] / F_psf[~mask]
F_deconv[mask] = F_blur[mask] 

unblur = np.fft.irfft2(F_deconv, s=blur_data.shape)

# Normalize for display
unblur -= unblur.min()
unblur /= unblur.max()

plt.figure(figsize=(8, 8))
plt.imshow(unblur, cmap='gray', origin='upper')
plt.title("Unblurred (Deconvolved) Image")
plt.show()