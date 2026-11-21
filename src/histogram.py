"""
Calculate and display the histogram of an image.

Usage:
    python histogram.py my_photo.jpg

"""

import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CHANNELS = [("Red", "red"), ("Green", "green"), ("Blue", "blue")]


def load_image(path, mode):
    image = Image.open(path)

    return np.array(image.convert(mode))


def calculate_histogram(pixels):
    histogram = [0] * 256 

    for value in pixels.flatten():
        histogram[value] += 1

    return histogram


def show_histograms(pixels, histograms, use_color):
    figure, (image_axis, histogram_axis) = plt.subplots(1, 2, figsize=(12, 5))

    # Left side: image
    if use_color:
        image_axis.imshow(pixels)
        image_axis.set_title("Color image")
    else:
        image_axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
        image_axis.set_title("Grayscale image")
    image_axis.axis("off")

    # Right side: one curve per histogram, all drawn on the same plot
    x_values = range(256)
    for label, plot_color, histogram in histograms:
        histogram_axis.fill_between(x_values, histogram, color=plot_color, alpha=0.3)
        histogram_axis.plot(x_values, histogram, color=plot_color, label=label, linewidth=1)

    histogram_axis.set_title("Color histogram" if use_color else "Histogram")
    histogram_axis.set_xlabel("Value (0 = darkest, 255 = brightest)")
    histogram_axis.set_ylabel("Number of pixels")
    histogram_axis.set_xlim(0, 255)
    histogram_axis.legend()

    plt.tight_layout()
    plt.show()

def calculate_all_histograms(pixels, use_color):
    if not use_color:
        histogram = calculate_histogram(pixels)
        return [("Gray", "black", histogram)]

    results = []
    for channel_index, (name, plot_color) in enumerate(CHANNELS):
        # pixels[:, :, 0] means: "all rows, all columns, only channel 0 (red)"
        channel_pixels = pixels[:, :, channel_index]

        histogram = calculate_histogram(channel_pixels)
        results.append((name, plot_color, histogram))

    return results

def histogram_equalize(pixels, histogram):
    cdf = np.cumsum(histogram)
    table = np.round(cdf * 255 / sum(histogram)).astype(np.uint8)

    return table[pixels]

def operation(path, mode): 
    if mode == "-c":
        pixels = load_image(path, "RGB")
    else:
        pixels = load_image(path, "L")

    return pixels

def main():
    if len(sys.argv) != 3:
        print("Usage: python histogram.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    use_color = sys.argv[2] == "-c"

    pixels = operation(image_path, sys.argv[2])
    histograms = calculate_all_histograms(pixels, use_color)

    if sys.argv[2] == "-e":
        histogram = calculate_histogram(pixels)
        pixels = histogram_equalize(pixels, histogram)
        histograms = calculate_all_histograms(pixels, False)

    height, width = pixels.shape[0], pixels.shape[1]
    print(f"Image size: {width} x {height} pixels")
    print(f"Mode: {'color (RGB)' if use_color else 'grayscale'}")

    for label, _, histogram in histograms:
        print(f"{label}: {sum(histogram)} pixels counted")

    show_histograms(pixels, histograms, use_color)


if __name__ == "__main__":
    main()
