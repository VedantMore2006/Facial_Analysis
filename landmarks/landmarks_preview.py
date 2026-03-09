"""
Landmarks Preview Visualization
Displays the selected landmarks used for facial analysis.
Can work with webcam feed or static image.
"""

import cv2
import mediapipe as mp
import numpy as np
from landmark_subset import LANDMARK_SUBSET
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'
# Color scheme for different facial regions (BGR format)
LANDMARK_COLORS = {
    # Face contour (1, 2, 13, 14, 152, 234, 454)
    'face': (255, 200, 100),  # Light blue
    # Eyes (33, 133, 145, 159, 263, 362, 374, 386)
    'eyes': (0, 255, 0),  # Green
    # Eye regions (50, 61, 63, 70, 78, 95, 280, 291, 296, 308, 324, 336)
    'eye_region': (0, 200, 255),  # Yellow
    # Mouth/lips (468, 472)
    'mouth': (0, 0, 255),  # Red
}


def get_landmark_color(idx):
    """Get color for specific landmark based on facial region."""
    if idx in [1, 2, 13, 14, 152, 234, 454]:
        return LANDMARK_COLORS['face']
    elif idx in [33, 133, 145, 159, 263, 362, 374, 386]:
        return LANDMARK_COLORS['eyes']
    elif idx in [468, 472]:
        return LANDMARK_COLORS['mouth']
    elif idx in [276, 280, 282, 283, 285, 293, 295, 300, 334]:  # Right eyebrow landmarks
        return (100, 255, 255)  # Bright yellow for eyebrow
    else:
        return LANDMARK_COLORS['eye_region']


class LandmarkVisualizer:
    """Visualizes the selected landmarks on face."""
    
    def __init__(self):
        """Initialize MediaPipe FaceMesh detector."""
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def draw_landmarks(self, frame, landmarks, show_labels=True, show_connections=False):
        """
        Draw the selected landmarks on frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Image frame (BGR format)
        landmarks : list
            MediaPipe landmark list (478 elements)
        show_labels : bool
            Whether to show landmark index labels
        show_connections : bool
            Whether to show connections between landmarks
        
        Returns
        -------
        np.ndarray
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Extract positions of selected landmarks
        landmark_positions = {}
        for idx in LANDMARK_SUBSET:
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmark_positions[idx] = (x, y)
        
        # Draw connections if requested (simple lines between nearby landmarks)
        if show_connections:
            sorted_indices = sorted(LANDMARK_SUBSET)
            for i in range(len(sorted_indices)):
                for j in range(i + 1, len(sorted_indices)):
                    idx1, idx2 = sorted_indices[i], sorted_indices[j]
                    # Only connect if indices are close (heuristic)
                    if abs(idx1 - idx2) <= 5:
                        cv2.line(annotated_frame, 
                                landmark_positions[idx1], 
                                landmark_positions[idx2], 
                                (150, 150, 150), 1)
        
        # Draw landmarks
        for idx in LANDMARK_SUBSET:
            x, y = landmark_positions[idx]
            color = get_landmark_color(idx)
            
            # Draw small circle for landmark (small dot)
            cv2.circle(annotated_frame, (x, y), 2, color, -1)
            cv2.circle(annotated_frame, (x, y), 2, (255, 255, 255), 1)  # White border
            
            # Draw label if requested
            if show_labels:
                # Background rectangle for text
                text = str(idx)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw semi-transparent background
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
        legend_height = 100
        legend = np.zeros((legend_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Title
        cv2.putText(legend, "Landmark Regions:", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend items
        y_offset = 50
        items = [
            (LANDMARK_COLORS['face'], "Face Contour"),
            (LANDMARK_COLORS['eyes'], "Eyes"),
            ((100, 255, 255), "Eyebrow"),
            (LANDMARK_COLORS['eye_region'], "Eye Region"),
            (LANDMARK_COLORS['mouth'], "Mouth"),
        ]
        
        for i, (color, label) in enumerate(items):
            x_offset = 10 + (i % 3) * 200
            y_pos = y_offset + (i // 3) * 30
            cv2.circle(legend, (x_offset, y_pos), 5, color, -1)
            cv2.circle(legend, (x_offset, y_pos), 5, (255, 255, 255), 1)
            cv2.putText(legend, label, (x_offset + 15, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Info text
        info_text = f"Total Landmarks Used: {len(LANDMARK_SUBSET)} out of 478"
        cv2.putText(legend, info_text, (10, legend_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine frame with legend
        return np.vstack([frame, legend])
    
    def visualize_webcam(self, show_labels=True, show_connections=False, show_legend=True):
        """
        Visualize landmarks using webcam feed.
        
        Parameters
        ----------
        show_labels : bool
            Whether to show landmark index labels
        show_connections : bool
            Whether to show connections between landmarks
        show_legend : bool
            Whether to show color legend
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam visualization started...")
        print("Press 'q' to quit")
        print("Press 'l' to toggle labels")
        print("Press 'c' to toggle connections")
        print("Press 's' to save screenshot")
        
        screenshot_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Draw landmarks if face detected
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                frame = self.draw_landmarks(frame, landmarks, show_labels, show_connections)
            else:
                # No face detected message
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add legend if requested
            if show_legend:
                frame = self.add_legend(frame)
            
            # Display frame
            cv2.imshow('Facial Landmarks Preview', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('c'):
                show_connections = not show_connections
                print(f"Connections: {'ON' if show_connections else 'OFF'}")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"landmark_preview_{screenshot_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def visualize_image(self, image_path, show_labels=True, show_connections=False, 
                       show_legend=True, save_output=None):
        """
        Visualize landmarks on a static image.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        show_labels : bool
            Whether to show landmark index labels
        show_connections : bool
            Whether to show connections between landmarks
        show_legend : bool
            Whether to show color legend
        save_output : str, optional
            Path to save annotated image
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            print("No face detected in image")
            return
        
        # Draw landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        annotated = self.draw_landmarks(frame, landmarks, show_labels, show_connections)
        
        # Add legend if requested
        if show_legend:
            annotated = self.add_legend(annotated)
        
        # Save if requested
        if save_output:
            cv2.imwrite(save_output, annotated)
            print(f"Saved annotated image to: {save_output}")
        
        # Display
        cv2.imshow('Facial Landmarks Preview', annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


def print_landmark_info():
    """Print information about the landmarks being used."""
    print("=" * 60)
    print("FACIAL LANDMARK SUBSET - PREVIEW")
    print("=" * 60)
    print(f"\nTotal landmarks used: {len(LANDMARK_SUBSET)} out of 478 MediaPipe landmarks\n")
    print("Landmark indices:")
    print(LANDMARK_SUBSET)
    print("\nRegion breakdown:")
    print("  - Face Contour: 7 landmarks")
    print("  - Eyes: 8 landmarks")
    print("  - Eyebrow: 9 landmarks")
    print("  - Eye Region: 12 landmarks")
    print("  - Mouth: 2 landmarks")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Visualize facial landmarks used in the analysis system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Webcam mode (no labels):
    python landmarks_preview.py
  
  Webcam mode with labels:
    python landmarks_preview.py --labels
  
  Image mode:
    python landmarks_preview.py --image photo.jpg --output result.png
  
  With connections visible:
    python landmarks_preview.py --connections
        '''
    )
    
    parser.add_argument('--image', '-i', type=str, 
                       help='Path to input image (if not provided, uses webcam)')
    parser.add_argument('--output', '-o', type=str, default='landmark_preview_output.png',
                       help='Output path for annotated image (default: landmark_preview_output.png)')
    parser.add_argument('--labels', '-l', action='store_true',
                       help='Show landmark index numbers (default: off)')
    parser.add_argument('--connections', '-c', action='store_true',
                       help='Show connections between landmarks (default: off)')
    parser.add_argument('--no-legend', action='store_true',
                       help='Hide the color legend (default: shown)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress landmark info output')
    
    args = parser.parse_args()
    
    # Print landmark info unless quiet mode
    if not args.quiet:
        print_landmark_info()
    
    visualizer = LandmarkVisualizer()
    
    # Check if image mode or webcam mode
    if args.image:
        # Image mode
        print(f"Processing image: {args.image}")
        visualizer.visualize_image(
            args.image, 
            show_labels=args.labels, 
            show_connections=args.connections,
            show_legend=not args.no_legend,
            save_output=args.output
        )
    else:
        # Webcam mode (default)
        print("Starting webcam mode...")
        print(f"Labels: {'ON' if args.labels else 'OFF'}")
        print(f"Connections: {'ON' if args.connections else 'OFF'}")
        visualizer.visualize_webcam(
            show_labels=args.labels, 
            show_connections=args.connections,
            show_legend=not args.no_legend
        )
