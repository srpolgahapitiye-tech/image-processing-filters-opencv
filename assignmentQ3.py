import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('runway.png')

if img is None:
    print("Error: Image not found!")
    exit()

## Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Calculate histogram
hist = np.zeros(256)

for pixel in gray.flatten():
    hist[pixel] += 1

## Normalize histogram
pdf = hist / np.sum(hist)

## Calculate cumulative distribution function
cdf = np.cumsum(pdf)

## Normalize CDF to the range [0, 255]
cdf_normalized = (cdf * 255).astype(np.uint8)

## Apply histogram equalization
equalized = cdf_normalized[gray]

## Display original and equalized images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(1,2,2)
plt.title("Histogram Equalized")
plt.imshow(equalized, cmap='gray')

plt.show()

## Display histograms
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Histogram")
plt.hist(gray.flatten(), bins=256)

plt.subplot(1,2,2)
plt.title("Equalized Histogram")
plt.hist(equalized.flatten(), bins=256)

plt.show()