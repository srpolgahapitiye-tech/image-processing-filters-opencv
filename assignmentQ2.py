import cv2
import numpy as np
import matplotlib.pyplot as plt

##Load image
img = cv2.imread('ronak-valobobhai.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

##Convert to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(lab)

##Normalize L channel
L_norm = L / 255.0

##Apply gamma correction
gamma = 0.5
L_gamma = np.power(L_norm, gamma)

##Scale back to [0,255] and convert to uint8
L_new = (L_gamma * 255).astype(np.uint8)
lab_new = cv2.merge((L_new, A, B))

## Convert back to image
img_result = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)

## Display results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Gamma Corrected (L channel)")
plt.imshow(img_result)

plt.show()

## Display histograms
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original L Histogram")
plt.hist(L.flatten(), bins=256)

plt.subplot(1,2,2)
plt.title("Corrected L Histogram")
plt.hist(L_new.flatten(), bins=256)

plt.show()