"""Language-controlled ball placement whose controller sees RGB-derived ball positions."""
from __future__ import annotations
import argparse
import numpy as np
import stage6_env  # noqa: F401
from stage6_skill import parse_ball_command, reset_nontrivial
from stage6_teacher import BALL_TASKS, BallPlaceTeacher, placement_success
from stage7_vision import BallVisionTracker
from vla_common import make_env


def execute_visual_place(env, observation, color, goal=None, max_steps=220,
                         render=False, image_mode="original", frame_callback=None,
                         motion_limit=1.0, track_until_descend=False,
                         release_height=0.025, settle_steps=0,
                         open_action=-1.0, open_steps=18,
                         transport_height=0.15):
    task = BALL_TASKS[color]
    # These privileged values are retained strictly for the final evaluator.
    initial_target = observation[task["key"]].copy()
    initial_reference = observation[task["reference_key"]].copy()
    tracker = BallVisionTracker(env.sim)
    detections = tracker.update(observation, image_mode)
    if any(detections[name] is None for name in ("red", "green")):
        return observation, color, False, 0, "vision_failed"
    # Both balls are static before contact. Lock the clean, unobstructed initial
    # estimates so the approaching gripper cannot bias a partially occluded mask.
    other_color = "green" if color == "red" else "red"
    tracker.freeze(other_color)
    if not track_until_descend:
        tracker.freeze(color)

    teacher = BallPlaceTeacher(
        env.action_dim, task["key"], task["reference_key"], goal=goal,
        max_motion=motion_limit, release_height=release_height,
        transport_height=transport_height,
    )
    teacher.settle_steps = int(settle_steps)
    teacher.open_action = float(open_action)
    teacher.open_steps = int(open_steps)
    for step in range(max_steps):
        # Replace privileged object observations with estimates obtained from RGB.
        visual_observation = dict(observation)
        visual_observation["cubeA_pos"] = tracker.require("red")
        visual_observation["cubeB_pos"] = tracker.require("green")
        action, status = teacher.act(visual_observation)
        observation, _, _, _ = env.step(action)
        if frame_callback is not None:
            frame_callback(observation)
        if render:
            env.render()
        tracker.update(observation, image_mode)
        if track_until_descend and status.phase == "descend":
            tracker.freeze(color)
        if status.phase == "done":
            break
    if goal is None:
        success = placement_success(observation, task, initial_target, initial_reference)
    else:
        target = np.asarray(observation[task["key"]])
        success = np.linalg.norm(target[:2] - np.asarray(goal)[:2]) < 0.035 and target[2] < 0.85
    return observation, color, success, step + 1, status.phase


def execute_visual_skill(env, observation, instruction, max_steps=220,
                         render=False, image_mode="original"):
    color = parse_ball_command(instruction)
    return execute_visual_place(
        env, observation, color, None, max_steps, render, image_mode
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="把绿色的球放到红色旁边")
    p.add_argument("--seed", type=int, default=7701)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--image-mode", choices=("original", "black"), default="original")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    env = make_env(renderer=args.render, seed=args.seed, env_name="ColorBallPlace")
    try:
        observation = reset_nontrivial(env)
        _, color, success, steps, phase = execute_visual_skill(
            env, observation, args.instruction, args.max_steps, args.render,
            args.image_mode,
        )
        print(f"instruction={args.instruction}")
        print(f"target={color} image={args.image_mode} success={success} "
              f"phase={phase} steps={steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
