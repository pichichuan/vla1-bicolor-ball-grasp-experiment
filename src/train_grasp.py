"""Train a visual-language action-chunk policy for cube grasping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import MiniVLA
from vla_common import build_vocabulary, encode_text


PICK_INSTRUCTIONS = (
    "pick up the cube",
    "lift the red block",
    "grasp and raise the cube",
)
WAIT_INSTRUCTIONS = (
    "wait",
    "do nothing",
    "stay still",
)
MAX_TOKENS = 8


def build_action_chunks(
    actions: np.ndarray, episode_ids: np.ndarray, chunk_size: int
) -> np.ndarray:
    chunks = np.empty((len(actions), chunk_size, actions.shape[1]), dtype=np.float32)
    for index in range(len(actions)):
        episode = episode_ids[index]
        episode_end = index + 1
        while episode_end < len(actions) and episode_ids[episode_end] == episode:
            episode_end += 1
        available = actions[index : min(index + chunk_size, episode_end)]
        chunks[index, : len(available)] = available
        chunks[index, len(available) :] = available[-1]
    return chunks


class GraspDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        states: np.ndarray,
        action_chunks: np.ndarray,
        base_indices: np.ndarray,
        vocabulary: dict[str, int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ):
        self.images = images
        self.states = states
        self.action_chunks = action_chunks
        self.base_indices = base_indices
        self.vocabulary = vocabulary
        self.state_mean = state_mean
        self.state_std = state_std

    def __len__(self) -> int:
        # Every observation has a pick and a counterfactual wait example.
        return len(self.base_indices) * 2

    def __getitem__(self, item: int):
        index = self.base_indices[item // 2]
        is_wait = item % 2 == 1
        phrases = WAIT_INSTRUCTIONS if is_wait else PICK_INSTRUCTIONS
        phrase = phrases[index % len(phrases)]
        target = (
            np.zeros_like(self.action_chunks[index])
            if is_wait
            else self.action_chunks[index]
        )
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(
            ((self.states[index] - self.state_mean) / self.state_std).astype(np.float32)
        )
        tokens = torch.from_numpy(encode_text(phrase, self.vocabulary, MAX_TOKENS))
        return image, tokens, state, torch.from_numpy(target.reshape(-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/grasp_demos.npz"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/grasp_vla.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(args.data) as data:
        images = data["images"]
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        episode_ids = data["episode_ids"]

    action_chunks = build_action_chunks(actions, episode_ids, args.chunk_size)
    vocabulary = build_vocabulary(list(PICK_INSTRUCTIONS + WAIT_INSTRUCTIONS))

    rng = np.random.default_rng(args.seed)
    episodes = np.unique(episode_ids)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(0.2 * len(episodes))))
    validation_episodes = set(episodes[:validation_count].tolist())
    validation_mask = np.asarray([episode in validation_episodes for episode in episode_ids])
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)

    state_mean = states[train_indices].mean(axis=0)
    state_std = states[train_indices].std(axis=0)
    state_std[state_std < 1e-5] = 1.0
    train_dataset = GraspDataset(
        images,
        states,
        action_chunks,
        train_indices,
        vocabulary,
        state_mean,
        state_std,
    )
    validation_dataset = GraspDataset(
        images,
        states,
        action_chunks,
        validation_indices,
        vocabulary,
        state_mean,
        state_std,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    action_dim = actions.shape[1]
    output_dim = args.chunk_size * action_dim
    model = MiniVLA(len(vocabulary), states.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    per_action_weight = torch.tensor(
        [2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 1.0], device=device
    ).repeat(args.chunk_size)
    best_validation_loss = float("inf")

    print(
        f"device={device}, train={len(train_dataset)}, validation={len(validation_dataset)}, "
        f"chunk={args.chunk_size}, parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for image, tokens, state, target in train_loader:
            image, tokens = image.to(device), tokens.to(device)
            state, target = state.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image, tokens, state)
            loss = ((prediction - target).square() * per_action_weight).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.shape[0]

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for image, tokens, state, target in validation_loader:
                image, tokens = image.to(device), tokens.to(device)
                state, target = state.to(device), target.to(device)
                prediction = model(image, tokens, state)
                loss = ((prediction - target).square() * per_action_weight).mean()
                validation_loss += loss.item() * image.shape[0]
        train_loss /= len(train_dataset)
        validation_loss /= len(validation_dataset)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabulary": vocabulary,
                    "max_tokens": MAX_TOKENS,
                    "state_dim": states.shape[1],
                    "action_dim": action_dim,
                    "chunk_size": args.chunk_size,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "validation_loss": validation_loss,
                },
                args.output,
            )

    print(f"best validation loss={best_validation_loss:.6f}")
    print(f"checkpoint saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
