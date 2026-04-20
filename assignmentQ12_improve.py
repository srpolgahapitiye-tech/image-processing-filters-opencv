import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load image
# =========================
img = cv2.imread("runway.png", cv2.IMREAD_GRAYSCALE)

# Convert to float
img = img.astype(np.float32)

# =========================
# Step 1: Log Transform
# =========================
log_img = np.log1p(img)

# =========================
# Step 2: FFT
# =========================
fft = np.fft.fft2(log_img)
fft_shift = np.fft.fftshift(fft)

# =========================
# Step 3: Create Homomorphic Filter
# =========================
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Create meshgrid
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

# Distance from center
D2 = (X - ccol)**2 + (Y - crow)**2

# Parameters (IMPORTANT)
gamma_l = 0.9   # suppress illumination
gamma_h = 1.3   # enhance reflectance
sigma = 150      # control cutoff (increase if too dark)

# Homomorphic filter
H = (gamma_h - gamma_l) * (1 - np.exp(-D2 / (2 * sigma**2))) + gamma_l

# =========================
# Step 4: Apply Filter
# =========================
filtered = fft_shift * H

# =========================
# Step 5: Inverse FFT
# =========================
ifft_shift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(ifft_shift)
img_back = np.real(img_back)

# =========================
# Step 6: Exponential Transform
# =========================
exp_img = np.exp(img_back) - 1

# =========================
# Step 7: Normalize to 0–255
# =========================
exp_img = exp_img - np.min(exp_img)
exp_img = exp_img / np.max(exp_img)
exp_img = (exp_img * 255).astype(np.uint8)

# =========================
# Display Results
# =========================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Homomorphic Filtered")
plt.imshow(exp_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()