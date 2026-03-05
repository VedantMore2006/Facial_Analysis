"""
Geometry Utilities

Provides scale-invariant distance computations for facial landmarks.
"""

import numpy as np


def distance(p1, p2):
	"""
	Euclidean distance between two 3D points.
	"""
	p1 = np.array(p1)
	p2 = np.array(p2)

	return np.linalg.norm(p1 - p2)


def inter_ocular_distance(landmarks):
	"""
	Compute inter-ocular distance using eye corners.

	Landmarks:
	33  = left eye corner
	263 = right eye corner
	"""
	left_eye = landmarks[33]
	right_eye = landmarks[263]

	return distance(left_eye, right_eye)


def normalized_distance(p1, p2, iod):
	"""
	Compute normalized landmark distance.

	normalized = distance / IOD
	"""
	if iod == 0:
		return 0

	return distance(p1, p2) / iod
