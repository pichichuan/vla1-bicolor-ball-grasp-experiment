"""Privileged state-machine teacher for the robosuite Lift task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from vla_common import make_env


@dataclass
class TeacherStatus:
    phase: str
    phase_step: int


class GraspTeacher:
    """Generate expert actions using simulator-only cube coordinates.

    The learned policy never receives cube_pos; only this demonstration teacher
    is privileged to use it.
    """

    def __init__(self, action_dim: int, object_key: str = "cube_pos"):
        self.action_dim = action_dim
        self.object_key = object_key
        self.phase = "approach"
        self.phase_step = 0
        self.lift_target: np.ndarray | None = None

    @staticmethod
    def _position_action(error: np.ndarray, gain: float = 18.0) -> np.ndarray:
        return np.clip(error * gain, -1.0, 1.0)

    def act(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, TeacherStatus]:
        eef = np.asarray(observation["robot0_eef_pos"])
        cube = np.asarray(observation[self.object_key])
        action = np.zeros(self.action_dim, dtype=np.float32)

        if self.phase == "approach":
            target = cube + np.array([0.0, 0.0, 0.10])
            error = target - eef
            action[:3] = self._position_action(error)
            action[-1] = -1.0
            if np.linalg.norm(error) < 0.012:
                self.phase = "descend"
                self.phase_step = 0

        elif self.phase == "descend":
            target = cube + np.array([0.0, 0.0, 0.005])
            error = target - eef
            action[:3] = self._position_action(error, gain=14.0)
            action[-1] = -1.0
            if np.linalg.norm(error) < 0.008:
                self.phase = "close"
                self.phase_step = 0

        elif self.phase == "close":
            target = cube + np.array([0.0, 0.0, 0.005])
            action[:3] = self._position_action(target - eef, gain=10.0)
            action[-1] = 1.0
            if self.phase_step >= 14:
                self.phase = "lift"
                self.phase_step = 0
                self.lift_target = eef + np.array([0.0, 0.0, 0.18])

        else:  # lift / done
            assert self.lift_target is not None
            error = self.lift_target - eef
            action[:3] = self._position_action(error, gain=12.0)
            action[-1] = 1.0
            if np.linalg.norm(error) < 0.015:
                self.phase = "done"

        status = TeacherStatus(self.phase, self.phase_step)
        self.phase_step += 1
        return action, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = make_env(renderer=args.render)
    successes = 0
    try:
        for episode in range(args.episodes):
            observation = env.reset()
            initial_height = float(observation["cube_pos"][2])
            teacher = GraspTeacher(env.action_dim)
            reward = 0.0
            for step in range(args.max_steps):
                action, status = teacher.act(observation)
                observation, reward, _, _ = env.step(action)
                if args.render:
                    env.render()
                if step % 10 == 0:
                    print(
                        f"episode={episode + 1} step={step:03d} phase={status.phase:<8} "
                        f"cube_z={observation['cube_pos'][2]:.3f}"
                    )
                if reward > 0 or status.phase == "done":
                    break
            lifted = float(observation["cube_pos"][2]) - initial_height
            success = bool(reward > 0 or lifted > 0.08)
            successes += int(success)
            print(
                f"episode={episode + 1} success={success} reward={reward:.3f} "
                f"lifted={lifted:.3f} steps={step + 1}"
            )
    finally:
        env.close()
    print(f"teacher success: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
