# Scaler module
"""
scaler.py

Purpose:
Apply deviation modeling:
Z-score + sigmoid bounding

Output:
Scaled value in [0,1]
"""

import numpy as np


def z_score(value, mu, sigma):
    return (value - mu) / sigma


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def scale_value(value, mu, sigma):
    z = z_score(value, mu, sigma)
    return sigmoid(z)