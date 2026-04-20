import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('ronak-valobobhai.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# OTSU THRESHOLDING
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
##mask = cv2.medianBlur(mask, 5) #improve mask by removing noise
print("Otsu Threshold Value:", ret)


# HISTOGRAM EQUALIZATION (manual)
def histogram_equalization(image):
    hist = np.zeros(256)

    for pixel in image.flatten():
        hist[pixel] += 1

    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    cdf_normalized = (cdf * 255).astype(np.uint8)

    return cdf_normalized[image]

equalized = histogram_equalization(gray)

# APPLY ONLY ON FOREGROUND
result = gray.copy()
result[mask == 255] = equalized[mask == 255]


# DISPLAY RESULTS
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original (Grayscale)")
plt.imshow(gray, cmap='gray')

plt.subplot(1,3,2)
plt.title("Foreground Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(1,3,3)
plt.title("Foreground Enhanced")
plt.imshow(result, cmap='gray')

plt.show()