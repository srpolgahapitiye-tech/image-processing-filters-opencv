import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('runway.png')

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize
img_norm = img / 255.0

## Gamma correction
## Gamma correction (γ = 0.5)
## Gamma correction (γ = 2)
def gamma_correction(image, gamma):
    return np.power(image, gamma)

gamma_05 = gamma_correction(img_norm, 0.5)
gamma_2 = gamma_correction(img_norm, 2)

## Contrast Stretching
def contrast_stretch(image, r1=0.2, r2=0.8):
    result = np.zeros_like(image)

    result[image < r1] = 0
    mask = (image >= r1) & (image <= r2)
    result[mask] = (image[mask] - r1) / (r2 - r1)
    result[image > r2] = 1

    return result

contrast_img = contrast_stretch(img_norm)

##Display Results

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(img_norm)

plt.subplot(2,2,2)
plt.title("Gamma 0.5")
plt.imshow(gamma_05)

plt.subplot(2,2,3)
plt.title("Gamma 2")
plt.imshow(gamma_2)

plt.subplot(2,2,4)
plt.title("Contrast Stretch")
plt.imshow(contrast_img)

plt.show()