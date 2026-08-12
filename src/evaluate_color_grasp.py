"""Test whether language selects the correct colored cube in identical scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from collect_color_grasp_demos import COLOR_TASKS
from evaluate_grasp import load_policy, predict_chunk
from vla_common import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/color_grasp_vla.pt"))
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--execute-chunk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def run_trial(env, model, checkpoint, observation, color, args, device):
    task = COLOR_TASKS[color]
    instruction = task["phrases"][0]
    initial_target = float(observation[task["key"]][2])
    initial_other = float(observation[task["other_key"]][2])
    step = 0
    while step < args.max_steps:
        chunk = predict_chunk(model, checkpoint, observation, instruction, device)
        for action in chunk[: args.execute_chunk]:
            observation, _, _, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            if args.render:
                env.render()
            target_lift = float(observation[task["key"]][2]) - initial_target
            other_lift = float(observation[task["other_key"]][2]) - initial_other
            if target_lift > 0.04 or other_lift > 0.04 or step >= args.max_steps:
                correct = target_lift > 0.04 and other_lift <= 0.04
                return correct, target_lift, other_lift, step
    return False, 0.0, 0.0, step


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(args.checkpoint, device)
    environments = {
        color: make_env(renderer=args.render, seed=args.seed, env_name="Stack")
        for color in COLOR_TASKS
    }
    correct = {"red": 0, "green": 0}
    try:
        for scene in range(args.scenes):
            observations = {color: env.reset() for color, env in environments.items()}
            paired_state = environments["red"].sim.get_state().flatten()
            environments["green"].sim.set_state_from_flattened(paired_state)
            environments["green"].sim.forward()
            observations["green"] = environments["green"]._get_observations(
                force_update=True
            )
            for color in COLOR_TASKS:
                passed, target_lift, wrong_lift, steps = run_trial(
                    environments[color],
                    model,
                    checkpoint,
                    observations[color],
                    color,
                    args,
                    device,
                )
                correct[color] += int(passed)
                print(
                    f"scene={scene + 1:02d} command={color:<5} correct={passed} "
                    f"target_lift={target_lift:+.3f} wrong_lift={wrong_lift:+.3f} steps={steps}"
                )
    finally:
        for env in environments.values():
            env.close()
    total = sum(correct.values())
    trials = args.scenes * 2
    print(
        f"red={correct['red']}/{args.scenes} green={correct['green']}/{args.scenes} "
        f"correct_target_rate={total}/{trials} ({total / trials:.1%})"
    )
    raise SystemExit(0 if total > 0 else 1)


if __name__ == "__main__":
    main()
