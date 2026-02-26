"""
plot_utils.py

Plot raw vs smoothed vs scaled signals.
"""

import matplotlib.pyplot as plt


def plot_signal(raw, smoothed, scaled, name):

    plt.figure(figsize=(10,5))

    plt.plot(raw, label="Raw")
    plt.plot(smoothed, label="Smoothed")
    plt.plot(scaled, label="Scaled")

    plt.title(name)
    plt.legend()
    plt.show()