import cv2
import numpy as np
import matplotlib.pyplot as plt

## Read the image and convert it to grayscale if use color image
img = cv2.imread('runway.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##Smooth image (Gaussian)
blur = cv2.GaussianBlur(gray, (5,5), 1)

##Compute gradients (DoG approximation using Sobel)
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

##Compute magnitude and direction of gradients
magnitude = np.sqrt(Gx**2 + Gy**2)
direction = np.arctan2(Gy, Gx)

## Normalize magnitude and direction to [0,255] for visualization
magnitude = np.uint8(255 * magnitude / np.max(magnitude))
direction = np.uint8(255 * (direction + np.pi) / (2*np.pi))

## Display results
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(2,2,2)
plt.title("Gx (Horizontal edges)")
plt.imshow(Gx, cmap='gray')

plt.subplot(2,2,3)
plt.title("Gy (Vertical edges)")
plt.imshow(Gy, cmap='gray')

plt.subplot(2,2,4)
plt.title("Magnitude")
plt.imshow(magnitude, cmap='gray')

plt.show()