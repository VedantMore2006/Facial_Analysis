import cv2
import os
import numpy as np
from datetime import datetime

os.environ['QT_QPA_PLATFORM'] = 'xcb'

from landmarks.mediapipe_detector import MediaPipeDetector
from landmarks.landmark_subset import extract_subset

# Complete MediaPipe Face Mesh - All 468 landmarks with all connections
FACE_MESH_CONNECTIONS = [
    # Lips (61 points) - outer and inner
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    (95, 88), (88, 178), (178, 87), (87, 10), (10, 398), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
    
    # Right eye (33 points)
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133), (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163), (163, 7),
    
    # Left eye (33 points)
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362), (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249),
    
    # Right eyebrow
    (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 193), (193, 7), (7, 163),
    
    # Left eyebrow
    (300, 293), (293, 334), (334, 296), (296, 336), (336, 285), (285, 413), (413, 441), (441, 265),
    
    # Face contour (34 points per side)
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    
    # Nose (13 points)
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19), (19, 94), (94, 3), (3, 248), (248, 456), (456, 457), (457, 123),
    
    # Nose top
    (4, 275), (275, 440), (440, 344), (344, 278), (278, 294), (294, 5),
    
    # Nose bridge
    (6, 238), (238, 241), (241, 235), (235, 288), (288, 464), (464, 457),
    
    # Additional face structure connections
    (10, 109), (109, 67), (67, 33), (33, 7), (7, 163), (163, 160), (160, 27), (27, 28), (28, 56), (56, 190), (190, 243), (243, 112), (112, 225), (225, 24), (24, 23), (23, 25),
    
    # Chin and lower jaw connections
    (200, 18), (18, 199), (199, 109), (108, 336), (336, 296), (296, 293), (293, 463), (463, 464), (464, 465), (465, 456), (456, 451), (451, 450), (450, 449), (449, 329), (329, 422), (422, 424), (424, 425), (425, 427), (427, 411), (411, 434), (434, 435), (435, 436), (436, 430), (430, 431), (431, 432), (432, 433), (433, 321), (321, 318), (318, 319), (319, 320), (320, 325), (325, 326), (326, 327), (327, 328), (328, 329), (329, 330), (330, 331), (331, 332), (332, 297), (297, 338), (338, 10),
    
    # Forehead connections
    (21, 162), (162, 127), (127, 234), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136), (136, 150), (150, 149), (149, 176), (176, 148), (148, 152), (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397), (397, 288), (288, 361), (361, 323), (323, 454), (454, 356), (356, 389), (389, 251), (251, 284), (284, 332), (332, 297), (297, 338), (338, 10),
    
    # Iris and pupils (additional detail)
    (468, 469), (469, 470), (470, 471), (471, 468),  # Right iris
    (472, 473), (473, 474), (474, 475), (475, 472),  # Left iris
]

# Configuration
CONFIG = {
    'dot_size': 2,
    'dot_color': (0, 255, 0),  # BGR format
    'draw_connections': True,
    'connection_color': (255, 0, 0),  # BGR format
    'line_thickness': 1,
    'show_fps': True,
}

SCREENSHOT_DIR = "./mesh_screenshots"


def create_screenshot_dir():
    """Create directory for screenshots if it doesn't exist."""
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
    return SCREENSHOT_DIR


def draw_mesh(frame, landmarks, config):
    """
    Draw facial mesh and landmarks on frame.
    
    Args:
        frame: Image frame
        landmarks: List of MediaPipe landmarks
        config: Configuration dictionary
    """
    h, w, _ = frame.shape
    
    # Draw connections
    if config['draw_connections'] and landmarks is not None:
        for start_idx, end_idx in FACE_MESH_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                start_pos = (int(start.x * w), int(start.y * h))
                end_pos = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_pos, end_pos, 
                        config['connection_color'], config['line_thickness'])
    
    # Draw dots
    if landmarks is not None:
        for landmark in landmarks:
            pos = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, pos, config['dot_size'], 
                      config['dot_color'], -1)
    
    return frame


def draw_controls_help(frame, config):
    """Draw control instructions on frame."""
    instructions = [
        "CONTROLS:",
        "+/- : Dot size",
        "C : Toggle connections",
        "S : Save screenshot",
        "D : Change dot color",
        "L : Change line color",
        "ESC : Exit"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display current settings
    settings_text = f"Dot Size: {config['dot_size']} | Connections: {'ON' if config['draw_connections'] else 'OFF'}"
    cv2.putText(frame, settings_text, (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


def draw_fps(frame, fps):
    """Draw FPS counter on frame."""
    if CONFIG['show_fps']:
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def save_screenshot(frame, landmarks, config):
    """Save screenshot with mesh."""
    create_screenshot_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOT_DIR, f"mesh_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✓ Screenshot saved: {filename}")
    print(f"  Landmarks detected: {len(landmarks) if landmarks else 0}")
    print(f"  Dot size: {config['dot_size']}")


def cycle_color(color_tuple):
    """Cycle through preset colors."""
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (255, 255, 255), # White
    ]
    current_idx = colors.index(color_tuple) if color_tuple in colors else 0
    return colors[(current_idx + 1) % len(colors)]


def main():
    detector = MediaPipeDetector()
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    create_screenshot_dir()
    
    # FPS calculation
    prev_frame_time = 0
    curr_frame_time = 0
    
    print("\n" + "="*50)
    print("FACIAL MESH VISUALIZATION")
    print("="*50)
    print("Press 'h' for controls help")
    print("Press 'ESC' to exit\n")
    
    show_help = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Calculate FPS
        curr_frame_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_frame_time
        
        # Detect landmarks
        landmarks = detector.detect(frame)
        
        # Draw mesh
        frame = draw_mesh(frame, landmarks, CONFIG)
        
        # Draw FPS
        draw_fps(frame, fps)
        
        # Show help
        if show_help:
            draw_controls_help(frame, CONFIG)
        
        # Display frame
        cv2.imshow("Facial Mesh Visualization", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nExiting...")
            break
        
        elif key == ord('+') or key == ord('='):  # Increase dot size
            CONFIG['dot_size'] = min(CONFIG['dot_size'] + 1, 10)
            print(f"Dot size: {CONFIG['dot_size']}")
        
        elif key == ord('-'):  # Decrease dot size
            CONFIG['dot_size'] = max(CONFIG['dot_size'] - 1, 1)
            print(f"Dot size: {CONFIG['dot_size']}")
        
        elif key == ord('c') or key == ord('C'):  # Toggle connections
            CONFIG['draw_connections'] = not CONFIG['draw_connections']
            print(f"Connections: {'ON' if CONFIG['draw_connections'] else 'OFF'}")
        
        elif key == ord('s') or key == ord('S'):  # Save screenshot
            if landmarks is not None:
                save_screenshot(frame, landmarks, CONFIG)
            else:
                print("No landmarks detected - cannot save")
        
        elif key == ord('d') or key == ord('D'):  # Change dot color
            CONFIG['dot_color'] = cycle_color(CONFIG['dot_color'])
            print(f"Dot color changed to: {CONFIG['dot_color']}")
        
        elif key == ord('l') or key == ord('L'):  # Change line color
            CONFIG['connection_color'] = cycle_color(CONFIG['connection_color'])
            print(f"Line color changed to: {CONFIG['connection_color']}")
        
        elif key == ord('h') or key == ord('H'):  # Toggle help
            show_help = not show_help
        
        elif key == ord('f') or key == ord('F'):  # Toggle FPS display
            CONFIG['show_fps'] = not CONFIG['show_fps']
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
