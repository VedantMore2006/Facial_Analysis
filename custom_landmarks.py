"""
Custom Landmark Subset for MediaPipe Face Mesh
Contains 29 specific landmarks for facial analysis.
"""

# Custom landmark indices
CUSTOM_LANDMARKS = [
    1, 2,           # Face contour
    13, 14,         # Face contour
    33,             # Left eye
    50,             # Eye region
    61, 63, 70,     # Eye/mouth region
    78, 95,         # Eye region
    107,            # Left eyebrow inner end
    133,            # Left eye
    145,            # Left eye
    152,            # Nose bridge
    159,            # Left eye
    234,            # Face contour
    263,            # Right eye
    280,            # Right eye region
    291, 296,       # Mouth region
    300,            # Right eyebrow outer end
    308, 324,       # Mouth region
    336,            # Right eyebrow inner end
    362,            # Right eye
    374,            # Right eye
    386,            # Right eye
    454,            # Face contour
    468, 472,       # Lips
]

def extract_custom_landmarks(landmarks):
    """
    Extract custom subset of landmarks from MediaPipe results.

    Parameters
    ----------
    landmarks : list
        MediaPipe landmark list (478 elements)

    Returns
    -------
    dict
        Dictionary mapping landmark index to (x, y, z) coordinates
    """
    subset = {}
    
    for idx in CUSTOM_LANDMARKS:
        lm = landmarks[idx]
        subset[idx] = (lm.x, lm.y, lm.z)
    
    return subset


def get_landmark_coordinates(landmarks, image_width, image_height):
    """
    Get pixel coordinates for custom landmarks.

    Parameters
    ----------
    landmarks : list
        MediaPipe landmark list (478 elements)
    image_width : int
        Width of the image
    image_height : int
        Height of the image

    Returns
    -------
    dict
        Dictionary mapping landmark index to (x_pixel, y_pixel) coordinates
    """
    pixel_coords = {}
    
    for idx in CUSTOM_LANDMARKS:
        lm = landmarks[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        pixel_coords[idx] = (x, y)
    
    return pixel_coords


if __name__ == "__main__":
    print("Custom MediaPipe Landmarks")
    print("=" * 50)
    print(f"Total landmarks: {len(CUSTOM_LANDMARKS)}")
    print(f"\nLandmark indices:")
    print(CUSTOM_LANDMARKS)
