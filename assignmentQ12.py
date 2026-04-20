import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("runway.png", cv2.IMREAD_GRAYSCALE)

# Convert to float
img = img.astype(np.float32)

# Step 1: Log transform
log_img = np.log1p(img)

# Step 2: FFT
fft = np.fft.fft2(log_img)
fft_shift = np.fft.fftshift(fft)

# Step 3: Create High-Pass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Gaussian High-pass
sigma = 30
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

distance = (X - ccol)**2 + (Y - crow)**2
H = 1 - np.exp(-distance / (2 * sigma**2))

# Step 4: Apply filter
filtered = fft_shift * H

# Step 5: Inverse FFT
ifft_shift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(ifft_shift)
img_back = np.abs(img_back)

# Step 6: Exponential
exp_img = np.expm1(img_back)

# Step 7: Normalize
exp_img = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
exp_img = np.uint8(exp_img)

# Display
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