from collections import deque


class FrameBuffer:
	def __init__(self, max_frames=150):
		self.buffer = deque(maxlen=max_frames)

	def add_frame(self, timestamp, landmarks, head_pose, features):
		frame_data = {
			"timestamp": timestamp,
			"landmarks": landmarks,
			"head_pose": head_pose,
			"features": features,
		}

		self.buffer.append(frame_data)

	def get_recent_frames(self, n):
		return list(self.buffer)[-n:]

	def get_all_frames(self):
		return list(self.buffer)

	def size(self):
		return len(self.buffer)
