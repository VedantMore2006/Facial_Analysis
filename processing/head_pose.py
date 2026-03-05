import numpy as np
import cv2


def estimate_head_pose(landmarks, frame_shape):
	"""
	Estimate head pose (yaw, pitch, roll).

	Parameters
	----------
	landmarks : dict
		Subset landmark dictionary {index: (x, y, z)}
	frame_shape : tuple
		Frame shape from OpenCV

	Returns
	-------
	tuple
		(yaw, pitch, roll)
	"""
	height, width = frame_shape[:2]

	# Use 6 anchor points for cv2.solvePnP (minimum required)
	image_points = np.array(
		[
			[landmarks[1][0] * width, landmarks[1][1] * height],      # Nose tip
			[landmarks[33][0] * width, landmarks[33][1] * height],    # Left eye
			[landmarks[263][0] * width, landmarks[263][1] * height],  # Right eye
			[landmarks[152][0] * width, landmarks[152][1] * height],  # Chin
			[landmarks[61][0] * width, landmarks[61][1] * height],    # Left mouth
			[landmarks[291][0] * width, landmarks[291][1] * height],  # Right mouth
		],
		dtype="double",
	)

	# 3D model points (approximate facial landmarks in mm)
	model_points = np.array(
		[
			(0.0, 0.0, 0.0),        # Nose tip
			(-30.0, 40.0, -30.0),   # Left eye
			(30.0, 40.0, -30.0),    # Right eye
			(0.0, -50.0, -30.0),    # Chin
			(-20.0, -20.0, -10.0),  # Left mouth
			(20.0, -20.0, -10.0),   # Right mouth
		],
		dtype="double",
	)

	focal_length = width
	center = (width / 2, height / 2)

	camera_matrix = np.array(
		[
			[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1],
		],
		dtype="double",
	)

	dist_coeffs = np.zeros((4, 1), dtype="double")

	success, rotation_vector, translation_vector = cv2.solvePnP(
		model_points,
		image_points,
		camera_matrix,
		dist_coeffs,
		flags=cv2.SOLVEPNP_ITERATIVE,
	)

	if not success:
		return 0.0, 0.0, 0.0

	# Use rotation vector angles directly
	# Each element of rotation_vector is angle * axis_unit_vector
	# Magnitude gives rotation angle, direction gives axis
	
	angle = np.linalg.norm(rotation_vector)
	
	if angle < 1e-6:
		# No rotation
		return 0.0, 0.0, 0.0
	
	# Normalize to get axis
	axis = rotation_vector / angle
	
	# Convert angle from radians to degrees
	angle_deg = angle * 180 / np.pi
	
	# Clamp unrealistic values
	angle_deg = np.clip(angle_deg, -90, 90)
	
	# Map axis components to yaw, pitch, roll
	# axis[0] = x-component (pitch axis)
	# axis[1] = y-component (yaw axis) 
	# axis[2] = z-component (roll axis)
	
	pitch = float(axis[0] * angle_deg)
	yaw = float(axis[1] * angle_deg)
	roll = float(axis[2] * angle_deg)

	return yaw, pitch, roll


def reject_unstable_frames(yaw, pitch):
	"""
	Check if frame head pose is unstable.

	Returns True when frame should be rejected.
	
	Thresholds:
	- Yaw: ±45° (horizontal head turn)
	- Pitch: ±30° (vertical head tilt)
	"""
	if abs(yaw) > 45:
		return True

	if abs(pitch) > 30:
		return True

	return False


def align_landmarks(landmarks):
	"""
	Align landmarks relative to nose center.
	"""
	nose = landmarks[1]
	aligned = {}

	for idx, (x, y, z) in landmarks.items():
		aligned[idx] = (
			x - nose[0],
			y - nose[1],
			z - nose[2],
		)

	return aligned
