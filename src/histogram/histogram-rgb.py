import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CHANNELS = [("Red", "red"), ("Green", "green"), ("Blue", "blue")]


def load_image(path):
    image = Image.open(path)

    return np.array(image.convert("RGB"))


def calculate_histogram(pixels):
    histogram = [0] * 256 

    for value in pixels.flatten():
        histogram[value] += 1

    return histogram


def show_histograms(pixels, histograms):
    figure, (image_axis, histogram_axis) = plt.subplots(1, 2, figsize=(12, 5))

    # left side: image
    image_axis.imshow(pixels)
    image_axis.set_title("Color image")
    image_axis.axis("off")

    # right side: histograms
    x_values = range(256)
    for label, plot_color, histogram in histograms:
        histogram_axis.fill_between(x_values, histogram, color=plot_color, alpha=0.3)
        histogram_axis.plot(x_values, histogram, color=plot_color, label=label, linewidth=1)

    histogram_axis.set_title("Color histogram")
    histogram_axis.set_xlabel("Value (0 = darkest, 255 = brightest)")
    histogram_axis.set_ylabel("Number of pixels")
    histogram_axis.set_xlim(0, 255)
    histogram_axis.legend()

    plt.tight_layout()
    plt.show()

def calculate_all_histograms(pixels):
    results = []
    for channel_index, (name, plot_color) in enumerate(CHANNELS):
        channel_pixels = pixels[:, :, channel_index]

        histogram = calculate_histogram(channel_pixels)
        results.append((name, plot_color, histogram))

    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    pixels = load_image(image_path)
    histograms = calculate_all_histograms(pixels)

    height, width = pixels.shape[0], pixels.shape[1]
    print(f"Image size: {width} x {height} pixels")
    print(f"Mode: color (RGB)")

    for label, _, histogram in histograms:
        print(f"{label}: {sum(histogram)} pixels counted")

    show_histograms(pixels, histograms)


if __name__ == "__main__":
    main()
