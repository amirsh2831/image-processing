import sys

import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SALT_PEPPER_AMOUNT = 0.05
SIZE = 3

def load_image(path):
    image = Image.open(path).convert("L")

    return np.array(image)


#
# def add_gaussian_noise(image, rng):
#     noise = rng.normal(loc=0, scale=GAUSSIAN_STRENGTH, size=image.shape)
#
#     return np.clip(image + noise, 0, 255)
#

def add_noise(image, rng):
    noisy = image.copy() 

    random_values = rng.random(image.shape)

    noisy[random_values < SALT_PEPPER_AMOUNT / 2] = 0          # pepper (black)
    noisy[random_values > 1 - SALT_PEPPER_AMOUNT / 2] = 255    # salt (white)

    return noisy

def apply_window_filter(image, size):
    height, width = image.shape
    border = size // 2  

    padded = np.pad(image, border, mode="edge")

    result = np.zeros((height, width))

    mask = np.ones((size, size)) / (size * size)  

    for y in range(height):
        for x in range(width):
            window = padded[y : y + size, x : x + size]

            result[y, x] = np.median(window )

    return result


def show_results(images_and_titles):
    figure, axes = plt.subplots(1, 3, figsize=(15, 10))

    for axis, (title, image) in zip(axes.ravel(), images_and_titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    image_path = sys.argv[1]

    original = load_image(image_path)

    rng = np.random.default_rng(seed=0)

    noisy = add_noise(original, rng)
    # noisy = add_gaussian_noise(original, rng)
    # noise_name = "Gaussian"

    median_filtered = apply_window_filter(noisy, SIZE)

    height, width = original.shape
    print(f"Image size: {width} x {height} pixels")
    print(f"Filter size: {SIZE}x{SIZE}")

    show_results([
        ("Original", original),
        (f"With noise", noisy),
        (f"Median filter {SIZE}x{SIZE}\n", median_filtered),
    ])


if __name__ == "__main__":
    main()
