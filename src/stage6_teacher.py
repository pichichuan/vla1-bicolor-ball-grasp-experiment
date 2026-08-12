"""Privileged state-machine teacher for placing one colored ball beside another."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
import stage6_env  # noqa: F401 - registers ColorBallPlace
from vla_common import make_env

BALL_TASKS = {
    "red": {
        "key": "cubeA_pos", "reference_key": "cubeB_pos",
        "phrases": ("place the red ball next to the green ball", "put red beside green"),
    },
    "green": {
        "key": "cubeB_pos", "reference_key": "cubeA_pos",
        "phrases": ("place the green ball next to the red ball", "put green beside red"),
    },
}

@dataclass
class PlaceStatus:
    phase: str
    step: int

def beside_goal(reference: np.ndarray) -> np.ndarray:
    # Place toward the table center, 9 cm from the reference.
    direction = -1.0 if reference[0] >= 0 else 1.0
    return np.array([reference[0] + direction * 0.09, reference[1], 0.818])

class BallPlaceTeacher:
    def __init__(self, action_dim: int, target_key: str, reference_key: str, goal=None,
                 max_motion=1.0, release_height=0.025, transport_height=0.15):
        self.action_dim = action_dim
        self.target_key = target_key
        self.reference_key = reference_key
        self.phase = "approach"
        self.phase_step = 0
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64).copy()
        self.max_motion = float(max_motion)
        self.release_height = float(release_height)
        self.transport_height = float(transport_height)
        self.settle_steps = 0
        self.open_action = -1.0
        self.open_steps = 18
        self.lift_z = None

    def motion(self, error, gain=16.0):
        return np.clip(
            np.asarray(error) * gain, -self.max_motion, self.max_motion
        )

    def act(self, observation):
        eef = np.asarray(observation["robot0_eef_pos"])
        target = np.asarray(observation[self.target_key])
        reference = np.asarray(observation[self.reference_key])
        if self.goal is None:
            self.goal = beside_goal(reference)
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.phase == "approach":
            error = target + np.array([0, 0, 0.10]) - eef
            action[:3], action[-1] = self.motion(error), -1.0
            if np.linalg.norm(error) < 0.012:
                self.phase, self.phase_step = "descend", 0
        elif self.phase == "descend":
            error = target + np.array([0, 0, 0.004]) - eef
            action[:3], action[-1] = self.motion(error, 13.0), -1.0
            if np.linalg.norm(error) < 0.008:
                self.phase, self.phase_step = "close", 0
        elif self.phase == "close":
            error = target + np.array([0, 0, 0.004]) - eef
            action[:3], action[-1] = self.motion(error, 9.0), 1.0
            if self.phase_step >= 16:
                self.phase, self.phase_step = "lift", 0
                self.lift_z = float(eef[2] + 0.16)
        elif self.phase == "lift":
            waypoint = np.array([eef[0], eef[1], self.lift_z])
            error = waypoint - eef
            action[:3], action[-1] = self.motion(error, 12.0), 1.0
            if abs(error[2]) < 0.012:
                self.phase, self.phase_step = "transport", 0
        elif self.phase == "transport":
            waypoint = self.goal + np.array([0, 0, self.transport_height])
            error = waypoint - eef
            action[:3], action[-1] = self.motion(error, 12.0), 1.0
            if np.linalg.norm(error) < 0.015:
                self.phase, self.phase_step = "lower", 0
        elif self.phase == "lower":
            waypoint = self.goal + np.array([0, 0, self.release_height])
            error = waypoint - eef
            action[:3], action[-1] = self.motion(error, 10.0), 1.0
            if np.linalg.norm(error) < 0.012:
                next_phase = "settle" if self.settle_steps else "open"
                self.phase, self.phase_step = next_phase, 0
        elif self.phase == "settle":
            waypoint = self.goal + np.array([0, 0, self.release_height])
            error = waypoint - eef
            action[:3], action[-1] = self.motion(error, 7.0), 1.0
            if self.phase_step >= self.settle_steps:
                self.phase, self.phase_step = "open", 0
        elif self.phase == "open":
            action[-1] = self.open_action
            if self.phase_step >= self.open_steps:
                self.phase, self.phase_step = "retreat", 0
        elif self.phase == "retreat":
            waypoint = self.goal + np.array([0, 0, 0.12])
            error = waypoint - eef
            action[:3], action[-1] = self.motion(error, 10.0), -1.0
            if np.linalg.norm(error) < 0.015:
                self.phase = "done"
        status = PlaceStatus(self.phase, self.phase_step)
        self.phase_step += 1
        return action, status

def placement_success(observation, task, initial_target, initial_reference):
    target = np.asarray(observation[task["key"]])
    reference = np.asarray(observation[task["reference_key"]])
    distance = float(np.linalg.norm(target[:2] - reference[:2]))
    moved = float(np.linalg.norm(target[:2] - initial_target[:2]))
    reference_moved = float(np.linalg.norm(reference[:2] - initial_reference[:2]))
    on_table = float(target[2]) < 0.85
    return 0.055 <= distance <= 0.125 and moved > 0.04 and reference_moved < 0.05 and on_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = make_env(renderer=args.render, seed=args.seed, env_name="ColorBallPlace")
    successes = 0
    try:
        for episode in range(args.episodes):
            for _ in range(100):
                observation = env.reset()
                if np.linalg.norm(observation["cubeA_pos"][:2] - observation["cubeB_pos"][:2]) >= 0.14:
                    break
            color = ("red", "green")[episode % 2]
            task = BALL_TASKS[color]
            initial_target = observation[task["key"]].copy()
            initial_reference = observation[task["reference_key"]].copy()
            teacher = BallPlaceTeacher(env.action_dim, task["key"], task["reference_key"])
            for step in range(args.max_steps):
                action, status = teacher.act(observation)
                observation, _, _, _ = env.step(action)
                if args.render:
                    env.render()
                if status.phase == "done":
                    break
            success = placement_success(observation, task, initial_target, initial_reference)
            successes += int(success)
            distance = np.linalg.norm(
                observation[task["key"]][:2] - observation[task["reference_key"]][:2]
            )
            moved = np.linalg.norm(observation[task["key"]][:2] - initial_target[:2])
            ref_moved = np.linalg.norm(observation[task["reference_key"]][:2] - initial_reference[:2])
            print(f"episode={episode+1:02d} color={color} success={success} phase={status.phase} distance={distance:.3f} moved={moved:.3f} ref_moved={ref_moved:.3f} steps={step+1}")
    finally:
        env.close()
    print(f"teacher_place_success={successes}/{args.episodes} ({successes/args.episodes:.1%})")

if __name__ == "__main__":
    main()
