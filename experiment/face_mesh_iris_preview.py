"""
Standalone Face Mesh + Iris preview (478 landmarks) for experiments.

This script is intentionally isolated from the main codebase.
It opens the webcam, runs MediaPipe Face Mesh with iris refinement,
and draws:
- Full mesh tessellation
- Face contours
- Iris connections

Runtime controls:
- q: Quit
- c: Cycle mesh color
- v: Cycle contour color
- i: Cycle iris color
- + or =: Increase line thickness
- -: Decrease line thickness
- r: Reset to default visual settings

Example:
    python experiment/face_mesh_iris_preview.py
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp

import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

Color = Tuple[int, int, int]  # OpenCV BGR


@dataclass
class PreviewConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 720

    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    thickness: int = 1
    circle_radius: int = 1

    mesh_color: Color = (80, 220, 100)
    contour_color: Color = (40, 180, 255)
    iris_color: Color = (255, 120, 30)
    
    show_connections: bool = True  # Toggle to show/hide mesh connections


COLOR_PALETTE: List[Color] = [
    (80, 220, 100),   # Green
    (255, 120, 30),   # Blue-ish orange in BGR
    (255, 255, 255),  # White
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 200, 255),    # Orange
    (220, 80, 220),   # Pink
]


class ColorCycler:
    def __init__(self, initial: Color):
        self.index = 0
        if initial in COLOR_PALETTE:
            self.index = COLOR_PALETTE.index(initial)

    def next(self) -> Color:
        self.index = (self.index + 1) % len(COLOR_PALETTE)
        return COLOR_PALETTE[self.index]


def draw_landmark_dots_only(frame, face_landmarks, cfg: PreviewConfig) -> None:
    """
    Draw only the individual landmark points without any connecting lines.
    Creates a dot-matrix visualization for easier exclusion decisions.
    """
    h, w, _ = frame.shape
    
    # Draw all 478 landmarks as individual dots
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), cfg.circle_radius, cfg.mesh_color, -1)


def put_overlay(frame, cfg: PreviewConfig) -> None:
    conn_state = "ON" if cfg.show_connections else "OFF (dots only)"
    lines = [
        "Controls: q=quit c=mesh-color v=contour-color i=iris-color e=toggle-connections +=thicker -=thinner r=reset",
        f"Thickness: {cfg.thickness} | Connections: {conn_state} | Mesh: {cfg.mesh_color} | Contour: {cfg.contour_color} | Iris: {cfg.iris_color}",
    ]

    y = 24
    for text in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += 22


def main() -> None:
    cfg = PreviewConfig()
    default_cfg = PreviewConfig()

    mesh_cycler = ColorCycler(cfg.mesh_color)
    contour_cycler = ColorCycler(cfg.contour_color)
    iris_cycler = ColorCycler(cfg.iris_color)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    mesh_spec = mp_drawing.DrawingSpec(
        color=cfg.mesh_color,
        thickness=cfg.thickness,
        circle_radius=cfg.circle_radius,
    )
    contour_spec = mp_drawing.DrawingSpec(
        color=cfg.contour_color,
        thickness=cfg.thickness,
        circle_radius=cfg.circle_radius,
    )
    iris_spec = mp_drawing.DrawingSpec(
        color=cfg.iris_color,
        thickness=cfg.thickness,
        circle_radius=cfg.circle_radius,
    )

    cap = cv2.VideoCapture(cfg.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device ID.")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=cfg.max_num_faces,
        refine_landmarks=True,  # Enables iris landmarks for 478 total points
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if cfg.show_connections:
                        # Full 478-point mesh connections
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mesh_spec,
                        )

                        # Face contours for easier landmark-group inspection
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=contour_spec,
                        )

                        # Iris connections (extra refine_landmarks points)
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=iris_spec,
                        )
                    else:
                        # Draw only individual landmark dots (matrix view)
                        draw_landmark_dots_only(frame, face_landmarks, cfg)

            put_overlay(frame, cfg)
            cv2.imshow("Face Mesh 478 Preview", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                cfg.mesh_color = mesh_cycler.next()
                mesh_spec.color = cfg.mesh_color
            if key == ord("v"):
                cfg.contour_color = contour_cycler.next()
                contour_spec.color = cfg.contour_color
            if key == ord("i"):
                cfg.iris_color = iris_cycler.next()
                iris_spec.color = cfg.iris_color
            if key == ord("e"):
                cfg.show_connections = not cfg.show_connections
            if key in (ord("+"), ord("=")):
                cfg.thickness = min(cfg.thickness + 1, 6)
                mesh_spec.thickness = cfg.thickness
                contour_spec.thickness = cfg.thickness
                iris_spec.thickness = cfg.thickness
            if key == ord("-"):
                cfg.thickness = max(cfg.thickness - 1, 1)
                mesh_spec.thickness = cfg.thickness
                contour_spec.thickness = cfg.thickness
                iris_spec.thickness = cfg.thickness
            if key == ord("r"):
                cfg = PreviewConfig(
                    camera_id=cfg.camera_id,
                    width=cfg.width,
                    height=cfg.height,
                    max_num_faces=cfg.max_num_faces,
                    min_detection_confidence=cfg.min_detection_confidence,
                    min_tracking_confidence=cfg.min_tracking_confidence,
                    thickness=default_cfg.thickness,
                    circle_radius=default_cfg.circle_radius,
                    mesh_color=default_cfg.mesh_color,
                    contour_color=default_cfg.contour_color,
                    iris_color=default_cfg.iris_color,
                )
                mesh_spec.color = cfg.mesh_color
                contour_spec.color = cfg.contour_color
                iris_spec.color = cfg.iris_color
                mesh_spec.thickness = cfg.thickness
                contour_spec.thickness = cfg.thickness
                iris_spec.thickness = cfg.thickness

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
