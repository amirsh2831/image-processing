
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_gray_image(path):
    image = Image.open(path).convert("L")
    return np.array(image, dtype=float)


def dft_matrix(n):
    k = np.arange(n).reshape(n, 1)  # 0, 1, 2, ... as a column
    m = np.arange(n).reshape(1, n)  # 0, 1, 2, ... as a row

    return np.exp(-2j * np.pi * k * m / n)


def calculate_dft(pixels):
    rows, columns = pixels.shape
    scale = np.sqrt(rows * columns)

    # matrix multiplication
    return dft_matrix(rows) @ pixels @ dft_matrix(columns) / scale


def magnitude_spectrum(dft):
    magnitude = np.abs(dft)
    centered = np.fft.fftshift(magnitude)

    return np.log(1 + centered) # log scaling


def show_dft(pixels, spectrum):
    figure, (image_axis, spectrum_axis) = plt.subplots(1, 2, figsize=(12, 5))

    image_axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    image_axis.set_title("Grayscale image")
    image_axis.axis("off")

    low, high = np.percentile(spectrum, [1, 99.5])
    spectrum_axis.imshow(spectrum, cmap="gray", vmin=low, vmax=high)

    spectrum_axis.set_title("Magnitude spectrum")
    spectrum_axis.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python dft.py <image_path>")
        sys.exit(1)

    pixels = load_gray_image(sys.argv[1])
    dft = calculate_dft(pixels)

    height, width = pixels.shape
    show_dft(pixels, magnitude_spectrum(dft))


if __name__ == "__main__":
    main()
