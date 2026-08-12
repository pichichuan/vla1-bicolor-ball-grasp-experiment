"""Reusable language-controlled ball placement skill."""
from __future__ import annotations
import argparse
import re
import numpy as np
import stage6_env  # noqa: F401
from stage6_teacher import BALL_TASKS, BallPlaceTeacher, placement_success
from vla_common import make_env

def parse_ball_command(text: str) -> str:
    normalized = text.lower().strip()
    chinese = re.search(r"把[^红绿]*(红色|绿色|红|绿)", normalized)
    if chinese:
        return "red" if chinese.group(1) in ("红色", "红") else "green"
    if normalized.startswith(("place ", "put ", "move ", "pick up ", "grasp ")):
        for token in normalized.split():
            if token in ("red", "green"):
                return token
    english = re.search(r"(?:place|put|move|pick up|grasp)\\s+(?:the\\s+)?(red|green)", normalized)
    if english:
        return english.group(1)
    raise ValueError("无法确定目标球；请明确说“把红色/绿色的球放到……旁边”")

def execute_skill(env, observation, instruction: str, max_steps=220, render=False):
    color = parse_ball_command(instruction)
    task = BALL_TASKS[color]
    initial_target = observation[task["key"]].copy()
    initial_reference = observation[task["reference_key"]].copy()
    teacher = BallPlaceTeacher(env.action_dim, task["key"], task["reference_key"])
    for step in range(max_steps):
        action, status = teacher.act(observation)
        observation, _, _, _ = env.step(action)
        if render:
            env.render()
        if status.phase == "done":
            break
    success = placement_success(observation, task, initial_target, initial_reference)
    return observation, color, success, step + 1

def reset_nontrivial(env):
    for _ in range(100):
        observation = env.reset()
        if np.linalg.norm(observation["cubeA_pos"][:2] - observation["cubeB_pos"][:2]) >= 0.14:
            return observation
    raise RuntimeError("Could not sample a non-trivial ball placement scene")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="把绿色的球放到红色旁边")
    p.add_argument("--seed", type=int, default=6701)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    env = make_env(renderer=args.render, seed=args.seed, env_name="ColorBallPlace")
    try:
        observation = reset_nontrivial(env)
        _, color, success, steps = execute_skill(
            env, observation, args.instruction, render=args.render
        )
        print(f"instruction={args.instruction}")
        print(f"parsed_target={color} success={success} steps={steps}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
