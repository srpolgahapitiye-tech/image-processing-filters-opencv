import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("q8images/im01.png")  # change if needed

if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# 1. Gaussian Filter (OpenCV)
# -------------------------------
gaussian = cv2.GaussianBlur(gray, (5,5), 1)

# -------------------------------
# 2. Bilateral Filter (OpenCV)
# -------------------------------
bilateral_cv = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

# -------------------------------
# 3. Manual Bilateral Filter
# -------------------------------
def bilateral_manual(image, d, sigma_s, sigma_r):
    padded = np.pad(image, d//2, mode='reflect')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            wp_total = 0
            filtered_pixel = 0

            for k in range(-d//2, d//2 + 1):
                for l in range(-d//2, d//2 + 1):

                    neighbor = padded[i + k + d//2, j + l + d//2]

                    # Spatial Gaussian
                    gs = np.exp(-(k**2 + l**2) / (2 * sigma_s**2))

                    # Intensity Gaussian
                    gr = np.exp(-((neighbor - image[i,j])**2) / (2 * sigma_r**2))

                    w = gs * gr

                    filtered_pixel += neighbor * w
                    wp_total += w

            result[i,j] = filtered_pixel / wp_total

    return np.uint8(result)

bilateral_manual_img = bilateral_manual(gray, d=5, sigma_s=50, sigma_r=50)

# -------------------------------
# Display
# -------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Gaussian")
plt.imshow(gaussian, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Bilateral (OpenCV)")
plt.imshow(bilateral_cv, cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("Bilateral (Manual)")
plt.imshow(bilateral_manual_img, cmap='gray')
plt.axis('off')

plt.show()