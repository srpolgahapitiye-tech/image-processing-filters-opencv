import numpy as np

## Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel

kernel_5 = gaussian_kernel(5, 2)

print(kernel_5)

## Visualize the Gaussian kernel in 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

kernel_51 = gaussian_kernel(51, 2)

x = np.arange(51)
y = np.arange(51)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, kernel_51)
ax.set_title("Gaussian Kernel (3D)")

plt.show()

## Apply Gaussian (Manual Convolution)
import cv2
# Load image
img = cv2.imread('runway.png')

if img is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_filter(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2

    output = np.zeros_like(image)

    padded = np.pad(image, k, mode='constant')

    for i in range(h):
        for j in range(w):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            output[i, j] = np.sum(region * kernel)

    return output

manual_blur = apply_filter(gray, kernel_5)

## Apply Gaussian (OpenCV)
opencv_blur = cv2.GaussianBlur(gray, (5,5), 2)

##Compare Results
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')

plt.subplot(1,3,2)
plt.title("Manual Gaussian")
plt.imshow(manual_blur, cmap='gray')

plt.subplot(1,3,3)
plt.title("OpenCV Gaussian")
plt.imshow(opencv_blur, cmap='gray')

plt.show()