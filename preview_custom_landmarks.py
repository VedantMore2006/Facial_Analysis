"""
Custom Landmarks Preview Visualization
Displays the custom 29 landmarks for facial analysis.
Works with webcam feed or static images.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from custom_landmarks import CUSTOM_LANDMARKS

os.environ['QT_QPA_PLATFORM'] = 'xcb'


# Color scheme for different facial regions (BGR format)
REGION_COLORS = {
    'face': (255, 200, 100),      # Light blue - face contour
    'eyes': (0, 255, 0),          # Green - eyes
    'eye_region': (0, 200, 255),  # Yellow - eye surrounding area
    'mouth': (0, 0, 255),         # Red - mouth/lips
    'nose': (255, 100, 255),      # Magenta - nose
}


def get_landmark_color(idx):
    """Get color for specific landmark based on facial region."""
    # Face contour
    if idx in [1, 2, 13, 14, 152, 234, 454]:
        return REGION_COLORS['face']
    # Eyes
    elif idx in [33, 133, 145, 159, 263, 362, 374, 386]:
        return REGION_COLORS['eyes']
    # Mouth/lips
    elif idx in [468, 472, 291, 296, 308, 324]:
        return REGION_COLORS['mouth']
    # Eye region
    else:
        return REGION_COLORS['eye_region']


class CustomLandmarkVisualizer:
    """Visualizes custom landmark subset on face."""
    
    def __init__(self):
        """Initialize MediaPipe FaceMesh detector."""
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def draw_landmarks(self, frame, landmarks, show_labels=False, dot_size=2):
        """
        Draw custom landmarks on frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Image frame (BGR format)
        landmarks : list
            MediaPipe landmark list (478 elements)
        show_labels : bool
            Whether to show landmark index labels
        dot_size : int
            Radius of landmark dots
        
        Returns
        -------
        np.ndarray
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Extract positions
        landmark_positions = {}
        for idx in CUSTOM_LANDMARKS:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmark_positions[idx] = (x, y)
        
        # Draw landmarks
        for idx in CUSTOM_LANDMARKS:
            x, y = landmark_positions[idx]
            color = get_landmark_color(idx)
            
            # Draw dot
            cv2.circle(annotated_frame, (x, y), dot_size, color, -1)
            cv2.circle(annotated_frame, (x, y), dot_size, (255, 255, 255), 1)
            
            # Draw label if requested
            if show_labels:
                text = str(idx)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Background for text
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, 
                             (x + 6, y - text_h - 4), 
                             (x + text_w + 10, y + 2), 
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # Draw text
                cv2.putText(annotated_frame, text, (x + 8, y - 2), 
                           font, font_scale, (255, 255, 255), thickness)
        
        return annotated_frame
    
    def add_legend(self, frame):
        """Add color legend to frame."""
        legend_height = 120
        legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Title
        cv2.putText(legend, "Custom Landmark Regions:", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend items
        y_offset = 55
        items = [
            (REGION_COLORS['face'], "Face Contour"),
            (REGION_COLORS['eyes'], "Eyes"),
            (REGION_COLORS['eye_region'], "Eye Region"),
            (REGION_COLORS['mouth'], "Mouth"),
        ]
        
        for i, (color, label) in enumerate(items):
            x_offset = 10 + (i % 2) * 250
            y_pos = y_offset + (i // 2) * 30
            cv2.circle(legend, (x_offset, y_pos), 4, color, -1)
            cv2.circle(legend, (x_offset, y_pos), 4, (255, 255, 255), 1)
            cv2.putText(legend, label, (x_offset + 15, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Info text
        info_text = f"Total Landmarks: {len(CUSTOM_LANDMARKS)} out of 478 MediaPipe landmarks"
        cv2.putText(legend, info_text, (10, legend_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine frame with legend
        return np.vstack([frame, legend])
    
    def visualize_webcam(self, show_labels=False, show_legend=True, dot_size=2):
        """
        Visualize landmarks using webcam feed.
        
        Parameters
        ----------
        show_labels : bool
            Whether to show landmark index labels
        show_legend : bool
            Whether to show color legend
        dot_size : int
            Radius of landmark dots
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Custom Landmarks Preview - Webcam Mode")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'l' - Toggle labels")
        print("  's' - Save screenshot")
        print("  '+' - Increase dot size")
        print("  '-' - Decrease dot size")
        
        screenshot_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Draw landmarks if face detected
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                frame = self.draw_landmarks(frame, landmarks, show_labels, dot_size)
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add legend
            if show_legend:
                frame = self.add_legend(frame)
            
            # Display
            cv2.imshow('Custom Landmarks Preview', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"custom_landmarks_{screenshot_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                dot_size = min(dot_size + 1, 10)
                print(f"Dot size: {dot_size}")
            elif key == ord('-') or key == ord('_'):
                dot_size = max(dot_size - 1, 1)
                print(f"Dot size: {dot_size}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def visualize_image(self, image_path, show_labels=False, show_legend=True, 
                       dot_size=2, save_output=None):
        """
        Visualize landmarks on a static image.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        show_labels : bool
            Whether to show landmark index labels
        show_legend : bool
            Whether to show color legend
        dot_size : int
            Radius of landmark dots
        save_output : str, optional
            Path to save annotated image
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Detect landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            print("No face detected in image")
            return
        
        # Draw landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        annotated = self.draw_landmarks(frame, landmarks, show_labels, dot_size)
        
        # Add legend
        if show_legend:
            annotated = self.add_legend(annotated)
        
        # Save if requested
        if save_output:
            cv2.imwrite(save_output, annotated)
            print(f"Saved: {save_output}")
        
        # Display
        cv2.imshow('Custom Landmarks Preview', annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def print_info():
    """Print landmark information."""
    print("=" * 60)
    print("CUSTOM MEDIAPIPE LANDMARK PREVIEW")
    print("=" * 60)
    print(f"\nTotal landmarks: {len(CUSTOM_LANDMARKS)}")
    print(f"\nLandmark indices:")
    print(CUSTOM_LANDMARKS)
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize custom facial landmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Webcam mode (no labels):
    python preview_custom_landmarks.py
  
  Webcam with labels:
    python preview_custom_landmarks.py --labels
  
  Image mode:
    python preview_custom_landmarks.py --image photo.jpg --output result.png
  
  Larger dots:
    python preview_custom_landmarks.py --dot-size 4
        '''
    )
    
    parser.add_argument('--image', '-i', type=str,
                       help='Path to input image (if not provided, uses webcam)')
    parser.add_argument('--output', '-o', type=str, default='custom_landmarks_output.png',
                       help='Output path for image mode')
    parser.add_argument('--labels', '-l', action='store_true',
                       help='Show landmark numbers')
    parser.add_argument('--no-legend', action='store_true',
                       help='Hide color legend')
    parser.add_argument('--dot-size', '-d', type=int, default=2,
                       help='Dot radius (default: 2)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress info output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_info()
    
    visualizer = CustomLandmarkVisualizer()
    
    if args.image:
        # Image mode
        print(f"Processing: {args.image}")
        visualizer.visualize_image(
            args.image,
            show_labels=args.labels,
            show_legend=not args.no_legend,
            dot_size=args.dot_size,
            save_output=args.output
        )
    else:
        # Webcam mode
        print("Starting webcam mode...")
        visualizer.visualize_webcam(
            show_labels=args.labels,
            show_legend=not args.no_legend,
            dot_size=args.dot_size
        )
