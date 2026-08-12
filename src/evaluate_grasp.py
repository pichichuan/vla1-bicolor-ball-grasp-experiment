"""Evaluate the learned visual grasp policy in fresh randomized Lift episodes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import ColorAttentionVLA, MiniVLA
from train_grasp import PICK_INSTRUCTIONS
from vla_common import encode_text, get_image, get_state, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/grasp_vla.pt"))
    parser.add_argument("--instruction", default=PICK_INSTRUCTIONS[0], choices=PICK_INSTRUCTIONS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--execute-chunk", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--image-mode", choices=("original", "black"), default="original")
    return parser.parse_args()


def load_policy(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    output_dim = checkpoint["action_dim"] * checkpoint["chunk_size"]
    model_class = (
        ColorAttentionVLA
        if checkpoint.get("model_type") == "color_attention"
        else MiniVLA
    )
    model = model_class(
        len(checkpoint["vocabulary"]), checkpoint["state_dim"], output_dim
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_chunk(model, checkpoint, observation, instruction, device, image_mode="original"):
    image_array = get_image(observation).copy()
    if image_mode == "black":
        image_array.fill(0)
    image = torch.from_numpy(image_array).permute(2, 0, 1).float()
    image = (image / 255.0).unsqueeze(0).to(device)
    state = get_state(observation)
    state = (state - checkpoint["state_mean"]) / checkpoint["state_std"]
    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    tokens = encode_text(
        instruction, checkpoint["vocabulary"], checkpoint["max_tokens"]
    )
    tokens = torch.from_numpy(tokens).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(image, tokens, state)[0]
    return output.reshape(checkpoint["chunk_size"], checkpoint["action_dim"]).cpu().numpy()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_policy(args.checkpoint, device)
    if not 1 <= args.execute_chunk <= checkpoint["chunk_size"]:
        raise ValueError("--execute-chunk must be within the trained chunk size")
    env = make_env(renderer=args.render, seed=args.seed)
    successes = 0
    try:
        for episode in range(args.episodes):
            observation = env.reset()
            initial_height = float(observation["cube_pos"][2])
            reward = 0.0
            step = 0
            while step < args.max_steps and reward <= 0:
                chunk = predict_chunk(
                    model,
                    checkpoint,
                    observation,
                    args.instruction,
                    device,
                    image_mode=args.image_mode,
                )
                for action in chunk[: args.execute_chunk]:
                    observation, reward, _, _ = env.step(np.clip(action, -1.0, 1.0))
                    step += 1
                    if args.render:
                        env.render()
                        time.sleep(1.0 / 20.0)
                    if reward > 0 or step >= args.max_steps:
                        break
            lifted = float(observation["cube_pos"][2]) - initial_height
            success = bool(reward > 0 or lifted > 0.08)
            successes += int(success)
            print(
                f"episode={episode + 1:02d} success={success} "
                f"reward={reward:.3f} lifted={lifted:.3f} steps={step}"
            )
            if args.render and args.hold_seconds > 0:
                deadline = time.perf_counter() + args.hold_seconds
                while time.perf_counter() < deadline:
                    env.render()
                    time.sleep(1.0 / 30.0)
    finally:
        env.close()
    rate = successes / args.episodes
    print(f"success_rate={successes}/{args.episodes} ({rate:.1%})")
    raise SystemExit(0 if successes > 0 else 1)


if __name__ == "__main__":
    main()
