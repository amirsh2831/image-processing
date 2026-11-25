import argparse

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# GAUSSIAN_STRENGTH = 25
SALT_PEPPER_AMOUNT = 0.05
SIZE = 3


def load_gray_image(path):
    image = Image.open(path).convert("L")

    return np.array(image)

# def add_gaussian_noise(image, rng):
#     noise = rng.normal(loc=0, scale=GAUSSIAN_STRENGTH, size=image.shape)
#
#     # np.clip keeps the result inside the valid range 0-255.
#     return np.clip(image + noise, 0, 255)


def add_salt_and_pepper_noise(image, rng):
    noisy = image.copy()

    random_values = rng.random(image.shape)

    noisy[random_values < SALT_PEPPER_AMOUNT / 2] = 0          # pepper (black)
    noisy[random_values > 1 - SALT_PEPPER_AMOUNT / 2] = 255    # salt (white)

    return noisy


def average_filter(image, size):
    height, width = image.shape
    border = size // 2  

    # add padding on the original image pixels
    padded = np.pad(image, border, mode="edge")

    result = np.zeros((height, width))

    mask = np.ones((size, size)) / (size * size)  

    for y in range(height):
        for x in range(width):
            window = padded[y : y + size, x : x + size]
            result[y, x] = (window * mask).sum()

    return result



def show_results(original, noisy, filtered, noise_name, size):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [original, noisy, filtered]
    titles = [
        "Original",
        "Noisy image",
        f"Averaging filter {size}x{size}",
    ]

    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    image_path = sys.argv[1]
    original = load_gray_image(image_path)

    rng = np.random.default_rng(seed=0)

    noisy = add_salt_and_pepper_noise(original, rng)
    noise_name = "salt-and-pepper"

    # noisy = add_gaussian_noise(original, rng)
    # noise_name = "Gaussian"

    filtered = average_filter(noisy, SIZE)


    height, width = original.shape
    print(f"Image size: {width} x {height} pixels")
    print(f"Noise: {noise_name}")
    print(f"Filter: averaging {SIZE}x{SIZE}")

    show_results(original, noisy, filtered, noise_name, SIZE)


if __name__ == "__main__":
    main()
