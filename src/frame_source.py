# ============================================================================
# FRAME SOURCE ABSTRACTION
# ============================================================================
# Purpose: Unified interface for frame acquisition from different sources
# 
# Design Pattern: Strategy Pattern
# - FrameSource: Abstract base interface
# - CameraSource: Live webcam implementation
# - VideoFileSource: Offline video file implementation
# 
# Benefits:
# - Pipeline remains source-agnostic
# - Easy to add new sources (e.g., RTSP streams, image sequences)
# - Feature engines unchanged regardless of input source
# ============================================================================

"""
frame_source.py

Purpose:
Abstraction for frame input sources.

Provides:
- Unified interface for webcam and video files
- Pluggable source implementations
- Source-agnostic pipeline operation
"""

import cv2
from abc import ABC, abstractmethod
from config import CameraConfig


class FrameSource(ABC):
    """
    Abstract base class for frame acquisition.
    
    All frame sources must implement:
    - read(): Get next frame
    - release(): Free resources
    - get_fps(): Get frames per second
    - is_realtime(): Indicate if source is live or recorded
    """
    
    @abstractmethod
    def read(self):
        """
        Capture next frame from source.
        
        Returns:
            Tuple of (success_flag, frame)
            - success_flag: Boolean indicating if frame was retrieved
            - frame: NumPy array containing image in BGR format, or None if failed
        """
        pass
    
    @abstractmethod
    def release(self):
        """
        Release hardware/file resources.
        
        Must be called when done to free resources.
        """
        pass
    
    @abstractmethod
    def get_fps(self):
        """
        Get frames per second of the source.
        
        Returns:
            Float representing frames per second
        """
        pass
    
    @abstractmethod
    def is_realtime(self):
        """
        Indicate if source is realtime (live) or recorded (file).
        
        Used by pipeline to determine timestamp calculation method:
        - Realtime: use system time
        - Recorded: use deterministic frame-based time
        
        Returns:
            Boolean: True if realtime, False if recorded
        """
        pass


class CameraSource(FrameSource):
    """
    Live webcam frame source.
    
    Wraps existing Camera logic using OpenCV VideoCapture.
    Retrieves frames in realtime from webcam device.
    """
    
    def __init__(self, device_id=None, width=None, height=None, fps=None):
        """
        Initialize camera with configured parameters.
        
        Args:
            device_id: Camera device index (defaults to CameraConfig.DEVICE_ID)
            width: Frame width in pixels (defaults to CameraConfig.WIDTH)
            height: Frame height in pixels (defaults to CameraConfig.HEIGHT)
            fps: Target frames per second (defaults to CameraConfig.FPS)
        """
        # Use config defaults if not specified
        self.device_id = device_id if device_id is not None else CameraConfig.DEVICE_ID
        self.width = width if width is not None else CameraConfig.WIDTH
        self.height = height if height is not None else CameraConfig.HEIGHT
        self.target_fps = fps if fps is not None else CameraConfig.FPS
        
        # Open camera device
        self.cap = cv2.VideoCapture(self.device_id)
        
        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
    
    def read(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple of (success_flag, frame)
        """
        return self.cap.read()
    
    def release(self):
        """
        Release camera hardware resources.
        """
        self.cap.release()
    
    def get_fps(self):
        """
        Get actual FPS value from camera.
        
        Note: May differ from configured FPS based on hardware.
        
        Returns:
            Float representing frames per second
        """
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def is_realtime(self):
        """
        Camera source is realtime.
        
        Returns:
            True (camera is live source)
        """
        return True


class VideoFileSource(FrameSource):
    """
    Offline video file frame source.
    
    Reads frames sequentially from a video file.
    Provides deterministic playback with frame-based timing.
    """
    
    def __init__(self, video_path):
        """
        Initialize video file reader.
        
        Args:
            video_path: Path to video file (e.g., 'data/recording.mp4')
        """
        self.video_path = video_path
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        
        # Verify file opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Extract FPS from video metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Extract total frame count
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video file loaded: {video_path}")
        print(f"   FPS: {self.fps}")
        print(f"   Total frames: {self.total_frames}")
        print(f"   Duration: {self.total_frames / self.fps:.2f} seconds")
    
    def read(self):
        """
        Read next frame from video file.
        
        Returns:
            Tuple of (success_flag, frame)
            - success_flag: False when end of file reached
        """
        return self.cap.read()
    
    def release(self):
        """
        Release video file resources.
        """
        self.cap.release()
    
    def get_fps(self):
        """
        Get FPS from video file metadata.
        
        Returns:
            Float representing frames per second
        """
        return self.fps
    
    def is_realtime(self):
        """
        Video file source is not realtime.
        
        Returns:
            False (video file is recorded source)
        """
        return False
