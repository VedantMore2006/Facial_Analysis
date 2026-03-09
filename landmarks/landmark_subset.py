"""
Landmark Subset Extraction
Reduces 478 MediaPipe landmarks to the selected subset (38 landmarks).
Includes additional eyebrow landmarks for better facial expression tracking.
"""


LANDMARK_SUBSET = [
	1, 2, 13, 14,
	33, 50,
	61, 63, 70,
	78, 95,
	133, 145,
	152, 159,
	234,
	263, 276, 280, 282, 283, 285, 293, 295, 300,  # Right eyebrow area
	291, 296,
	308, 324,
	334, 336,
	362, 374, 386,
	454,
	468, 472,
]


def extract_subset(landmarks):
	"""
	Extract subset of landmarks.

	Parameters
	----------
	landmarks : list
		MediaPipe landmark list (478 elements)

	Returns
	-------
	dict
		{landmark_index: (x, y, z)}
	"""
	subset = {}

	for idx in LANDMARK_SUBSET:
		lm = landmarks[idx]
		subset[idx] = (lm.x, lm.y, lm.z)

	return subset
