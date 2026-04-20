import cv2
import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS
def zoom_nearest(img, new_h, new_w):
    h, w = img.shape
    scale_x = new_h / h
    scale_y = new_w / w

    zoomed = np.zeros((new_h, new_w), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            x = int(i / scale_x)
            y = int(j / scale_y)
            zoomed[i, j] = img[x, y]

    return zoomed


def zoom_bilinear(img, new_h, new_w):
    h, w = img.shape
    scale_x = new_h / h
    scale_y = new_w / w

    zoomed = np.zeros((new_h, new_w), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):

            x = i / scale_x
            y = j / scale_y

            x1 = int(x)
            y1 = int(y)

            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)

            dx = x - x1
            dy = y - y1

            value = (1 - dx)*(1 - dy)*img[x1, y1] + \
                    dx*(1 - dy)*img[x2, y1] + \
                    (1 - dx)*dy*img[x1, y2] + \
                    dx*dy*img[x2, y2]

            zoomed[i, j] = value

    return zoomed.astype(np.uint8)


def compute_ssd(img1, img2):
    return np.sum((img1.astype(np.float32) - img2.astype(np.float32))**2)


# IMAGE PAIRS - SMALL TO LARGE (remove comments to test more pairs)
pairs = [
    #("q8images/im01small.png", "q8images/im01.png"),
    #("q8images/im02small.png", "q8images/im02.png"),
    #("q8images/im03small.png", "q8images/im03.png"),
    #("q8images/taylor_small.jpg", "q8images/taylor.jpg"),
    ("q8images/taylor_very_small.jpg", "q8images/taylor.jpg")
]

# -----------------------------
# PROCESS ALL
# -----------------------------
for small_path, large_path in pairs:

    small = cv2.imread(small_path, 0)
    large = cv2.imread(large_path, 0)

    if small is None or large is None:
        print(f"Error loading {small_path} or {large_path}")
        continue

    h, w = large.shape

    # Zoom
    nn = zoom_nearest(small, h, w)
    bl = zoom_bilinear(small, h, w)

    # SSD
    ssd_nn = compute_ssd(large, nn)
    ssd_bl = compute_ssd(large, bl)

    print(f"\n=== {small_path} → {large_path} ===")
    print("SSD (Nearest):", ssd_nn)
    print("SSD (Bilinear):", ssd_bl)

    # Show
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(large, cmap='gray')

    plt.subplot(1,3,2)
    plt.title("Nearest")
    plt.imshow(nn, cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Bilinear")
    plt.imshow(bl, cmap='gray')

    plt.show()