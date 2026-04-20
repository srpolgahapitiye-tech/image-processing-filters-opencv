import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("q8images/im01.png")  # change if needed

if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur (important for noise removal)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Canny edge detection
edges = cv2.Canny(blur, 50, 150)   # thresholds can be tuned

# Display results
plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Blurred")
plt.imshow(blur, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()