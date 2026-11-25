import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

LAPLACIAN = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]], dtype=float)

# SHARPEN_AMOUNT = 1.0


def load_image(path):
    image = Image.open(path).convert("L")

    return np.array(image)


def apply_mask(image, mask):
    height, width = image.shape
    size = 3
    border = size // 2

    padded = np.pad(image, border, mode="edge")

    result = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            window = padded[y : y + size, x : x + size]

            result[y, x] = (window * mask).sum()

    return result


def show_results(original, high_pass):
    figure, (original_axis, high_pass_axis) = plt.subplots(1, 2, figsize=(15, 5))

    original_axis.imshow(original, cmap="gray", vmin=0, vmax=255)
    original_axis.set_title("Original")

    edge_strength = np.abs(high_pass)
    high_pass_axis.imshow(edge_strength, cmap="gray", vmin=0, vmax=np.percentile(edge_strength, 99))
    high_pass_axis.set_title("High-pass result (Laplacian)")


    for axis in (original_axis, high_pass_axis):
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python laplacian.py <image_path>")
        sys.exit(1)

    original = load_image(sys.argv[1])

    high_pass = apply_mask(original, LAPLACIAN)

    # sharpened = np.clip(original + SHARPEN_AMOUNT * high_pass, 0, 255)

    height, width = original.shape
    print(f"Image size: {width} x {height} pixels")

    show_results(original, high_pass)


if __name__ == "__main__":
    main()
