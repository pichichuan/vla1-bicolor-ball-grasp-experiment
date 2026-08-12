"""RGB color detection, table-plane unprojection, and short-term ball tracking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import robosuite.utils.transform_utils as transform_utils
from vla_common import CAMERA_NAME, IMAGE_SIZE, get_image

BALL_CENTER_Z = 0.818
# Empirical pinhole / visible-sphere centroid bias; calibrated by the evaluator.
XY_BIAS = np.array([-0.0160, -0.0025], dtype=np.float64)

@dataclass
class Detection:
    color: str
    pixel: np.ndarray
    world: np.ndarray
    pixels: int

def world_to_camera_matrix(sim):
    camera_id = sim.model.camera_name2id(CAMERA_NAME)
    fovy = sim.model.cam_fovy[camera_id]
    focal = 0.5 * IMAGE_SIZE / np.tan(fovy * np.pi / 360)
    intrinsic = np.array(
        [[focal, 0, IMAGE_SIZE / 2], [0, focal, IMAGE_SIZE / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    pose = transform_utils.make_pose(
        sim.data.cam_xpos[camera_id],
        sim.data.cam_xmat[camera_id].reshape(3, 3),
    )
    camera_pose = pose @ np.diag([1.0, -1.0, -1.0, 1.0])
    expanded = np.eye(4)
    expanded[:3, :3] = intrinsic
    return expanded @ transform_utils.pose_inv(camera_pose)

def pixel_to_table(pixel, matrix, z=BALL_CENTER_Z):
    # For fixed z, the 3D projection becomes a 3x3 planar homography.
    homography = np.column_stack(
        (matrix[:3, 0], matrix[:3, 1], z * matrix[:3, 2] + matrix[:3, 3])
    )
    row, column = pixel
    # This project uses robosuite's OpenGL image convention (bottom row first),
    # while the pinhole matrix follows OpenCV's top-row-first convention.
    row = IMAGE_SIZE - 1 - row
    world_plane = np.linalg.solve(homography, np.array([column, row, 1.0]))
    world_plane /= world_plane[2]
    return np.array([world_plane[0], world_plane[1], z], dtype=np.float64)

def color_mask(image, color):
    rgb = image.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "red":
        return (red > 90) & (red > green * 1.35) & (red > blue * 1.25)
    return (green > 70) & (green > red * 1.20) & (green > blue * 1.15)

def detect_ball(image, color, matrix):
    mask = color_mask(image, color)
    rows, columns = np.nonzero(mask)
    if len(rows) < 4:
        return None
    # Median is robust to a few isolated colored pixels.
    pixel = np.array([np.median(rows), np.median(columns)], dtype=np.float64)
    world = pixel_to_table(pixel, matrix)
    world[:2] -= XY_BIAS
    return Detection(color, pixel, world, int(len(rows)))

class BallVisionTracker:
    def __init__(self, sim, smoothing=0.65):
        self.matrix = world_to_camera_matrix(sim)
        self.smoothing = smoothing
        self.positions = {}
        self.frozen = set()

    def freeze(self, color):
        self.frozen.add(color)

    def update(self, observation, image_mode="original"):
        image = get_image(observation)
        if image_mode == "black":
            image = np.zeros_like(image)
        elif image_mode != "original":
            raise ValueError(f"Unknown image mode: {image_mode}")
        detections = {}
        for color in ("red", "green"):
            detection = detect_ball(image, color, self.matrix)
            detections[color] = detection
            if color in self.frozen or detection is None:
                continue
            if color in self.positions:
                self.positions[color] = (
                    self.smoothing * detection.world
                    + (1.0 - self.smoothing) * self.positions[color]
                )
            else:
                self.positions[color] = detection.world.copy()
        return detections

    def require(self, color):
        if color not in self.positions:
            raise RuntimeError(f"No visual position available for {color} ball")
        return self.positions[color].copy()
