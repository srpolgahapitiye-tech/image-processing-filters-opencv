import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('emma.jpeg')  # change if needed

if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float (important for negative values)
gray_float = np.float32(gray)

# Apply Laplacian
laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)

# Sharpen image
alpha = 1.0
sharpened = gray_float - alpha * laplacian

# Normalize to 0–255
sharpened = np.clip(sharpened, 0, 255)
sharpened = np.uint8(sharpened)

# Display
plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Laplacian")
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Sharpened")
plt.imshow(sharpened, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()