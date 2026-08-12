"""Shared environment and language helpers for the mini-VLA tutorial."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import robosuite as suite


CAMERA_NAME = "agentview"
IMAGE_SIZE = 96


@dataclass(frozen=True)
class Command:
    name: str
    phrases: tuple[str, ...]
    axis: int
    sign: float


COMMANDS = (
    Command("left", ("move the gripper left", "go left"), 0, -1.0),
    Command("right", ("move the gripper right", "go right"), 0, 1.0),
    Command("forward", ("move the gripper forward", "go forward"), 1, 1.0),
    Command("backward", ("move the gripper backward", "go backward"), 1, -1.0),
    Command("up", ("move the gripper up", "go up"), 2, 1.0),
    Command("down", ("move the gripper down", "go down"), 2, -1.0),
)

PHRASE_TO_COMMAND = {
    phrase: command
    for command in COMMANDS
    for phrase in command.phrases
}


def make_env(*, renderer: bool, seed: int = 7, env_name: str = "Lift"):
    """Build the same camera-enabled Panda environment for every stage."""
    np.random.seed(seed)
    return suite.make(
        env_name=env_name,
        robots="Panda",
        has_renderer=renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERA_NAME,
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
        control_freq=20,
        horizon=200,
        ignore_done=True,
        hard_reset=False,
    )


def get_image(observation: dict[str, np.ndarray]) -> np.ndarray:
    image = observation[f"{CAMERA_NAME}_image"]
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected camera image shape: {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def get_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Return stable position observations, excluding velocity and acceleration."""
    keys = (
        "robot0_joint_pos",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in keys if key not in observation]
    if missing:
        available = ", ".join(sorted(observation))
        raise KeyError(f"Missing {missing}. Available observations: {available}")
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32) for key in keys]
    )


def teacher_action(action_dim: int, command: Command, magnitude: float = 0.20) -> np.ndarray:
    """A transparent expert: map a language direction to Cartesian motion."""
    action = np.zeros(action_dim, dtype=np.float32)
    action[command.axis] = command.sign * magnitude
    return action


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(phrases: list[str] | tuple[str, ...]) -> dict[str, int]:
    words = sorted({word for phrase in phrases for word in tokenize(phrase)})
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode_text(text: str, vocabulary: dict[str, int], max_tokens: int) -> np.ndarray:
    ids = [vocabulary.get(word, vocabulary["<unk>"]) for word in tokenize(text)]
    ids = ids[:max_tokens]
    ids.extend([vocabulary["<pad>"]] * (max_tokens - len(ids)))
    return np.asarray(ids, dtype=np.int64)
