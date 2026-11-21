import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    image = Image.open(path)

    return np.array(image.convert("L")) # L for grayscale

def calculate_histogram(pixels):
    histogram = [0] * 256 

    for value in pixels.flatten():
        histogram[value] += 1

    return histogram

def show_histogram(pixels, histogram):
    figure, (image_axis, histogram_axis) = plt.subplots(1, 2, figsize=(12, 5))

    # left side: image
    image_axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    image_axis.set_title("Grayscale image")
    image_axis.axis("off")

    x_values = range(256)
    # for label, plot_color
    histogram_axis.fill_between(x_values, histogram, color="gray", alpha=0.3)
    histogram_axis.plot(x_values, histogram, color="gray", linewidth=1)

    histogram_axis.set_title( "Histogram")
    histogram_axis.set_xlabel("Value (0 = darkest, 255 = brightest)")
    histogram_axis.set_ylabel("Number of pixels")
    histogram_axis.set_xlim(0, 255)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    pixels = load_image(image_path)
    histogram = calculate_histogram(pixels)

    height, width = pixels.shape[0], pixels.shape[1]
    print(f"Image size: {width} x {height} pixels")


    show_histogram(pixels, histogram)


if __name__ == "__main__":
    main()
