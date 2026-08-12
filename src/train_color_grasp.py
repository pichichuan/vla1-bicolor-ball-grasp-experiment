"""Train a color-conditioned VLA on paired red and green grasp trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import ColorAttentionVLA, MiniVLA
from collect_color_grasp_demos import COLOR_TASKS
from train_grasp import MAX_TOKENS, build_action_chunks
from vla_common import build_vocabulary, encode_text


class ColorGraspDataset(Dataset):
    def __init__(
        self,
        images,
        states,
        chunks,
        instructions,
        indices,
        vocabulary,
        state_mean,
        state_std,
    ):
        self.images = images
        self.states = states
        self.chunks = chunks
        self.instructions = instructions
        self.indices = indices
        self.vocabulary = vocabulary
        self.state_mean = state_mean
        self.state_std = state_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(
            ((self.states[index] - self.state_mean) / self.state_std).astype(np.float32)
        )
        tokens = torch.from_numpy(
            encode_text(self.instructions[index], self.vocabulary, MAX_TOKENS)
        )
        target = torch.from_numpy(self.chunks[index].reshape(-1))
        return image, tokens, state, target


class ColorSelectionDataset(Dataset):
    def __init__(self, images, states, target_actions, vocabulary, state_mean, state_std):
        self.images = images
        self.states = states
        self.target_actions = target_actions
        self.vocabulary = vocabulary
        self.state_mean = state_mean
        self.state_std = state_std

    def __len__(self):
        return len(self.images) * 2

    def __getitem__(self, item):
        scene = item // 2
        color_index = item % 2
        color = ("red", "green")[color_index]
        phrase = COLOR_TASKS[color]["phrases"][0]
        image = torch.from_numpy(self.images[scene].copy()).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(
            ((self.states[scene] - self.state_mean) / self.state_std).astype(np.float32)
        )
        tokens = torch.from_numpy(encode_text(phrase, self.vocabulary, MAX_TOKENS))
        target = torch.from_numpy(self.target_actions[scene, color_index].astype(np.float32))
        return image, tokens, state, target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/color_grasp_demos.npz"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/color_grasp_vla.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--selection-data",
        type=Path,
        default=Path("data/color_selection_demos.npz"),
    )
    parser.add_argument("--selection-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--initial-oversample",
        type=int,
        default=1,
        help="Repeat paired step-0 samples to force language-conditioned target selection.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--dagger-data", type=Path, action="append", default=[],
        help="May be repeated to merge multiple DAgger rounds.",
    )
    parser.add_argument("--model-type", choices=("mini", "color_attention"), default="mini")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.data) as data:
        images = data["images"]
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        instructions = data["instructions"].astype(str)
        episode_ids = data["episode_ids"]
        scene_ids = data["scene_ids"]
        step_ids = data["step_ids"]

    for dagger_path in args.dagger_data:
        with np.load(dagger_path) as dagger:
            episode_offset = int(episode_ids.max()) + 1
            scene_offset = int(scene_ids.max()) + 1
            images = np.concatenate((images, dagger["images"]))
            states = np.concatenate((states, dagger["states"].astype(np.float32)))
            actions = np.concatenate((actions, dagger["actions"].astype(np.float32)))
            instructions = np.concatenate((instructions, dagger["instructions"].astype(str)))
            episode_ids = np.concatenate((episode_ids, dagger["episode_ids"] + episode_offset))
            scene_ids = np.concatenate((scene_ids, dagger["scene_ids"] + scene_offset))
            step_ids = np.concatenate((step_ids, dagger["step_ids"]))
        print(f"merged DAgger corrections from {dagger_path.resolve()}")

    chunks = build_action_chunks(actions, episode_ids, args.chunk_size)
    vocabulary = build_vocabulary(instructions.tolist())
    rng = np.random.default_rng(args.seed)
    scenes = np.unique(scene_ids)
    rng.shuffle(scenes)
    validation_scenes = set(scenes[: max(1, round(0.2 * len(scenes)))].tolist())
    validation_mask = np.asarray([scene in validation_scenes for scene in scene_ids])
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    paired_initial_indices = train_indices[step_ids[train_indices] == 0]
    if args.initial_oversample > 1:
        train_indices = np.concatenate(
            [
                train_indices,
                np.repeat(paired_initial_indices, args.initial_oversample - 1),
            ]
        )
    state_mean = states[train_indices].mean(axis=0)
    state_std = states[train_indices].std(axis=0)
    state_std[state_std < 1e-5] = 1.0

    datasets = [
        ColorGraspDataset(
            images,
            states,
            chunks,
            instructions,
            indices,
            vocabulary,
            state_mean,
            state_std,
        )
        for indices in (train_indices, validation_indices)
    ]
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets[1], batch_size=args.batch_size)
    selection_loader = None
    if args.selection_data.exists():
        with np.load(args.selection_data) as selection_data:
            selection_dataset = ColorSelectionDataset(
                selection_data["images"],
                selection_data["states"].astype(np.float32),
                selection_data["target_actions"].astype(np.float32),
                vocabulary,
                state_mean,
                state_std,
            )
        selection_loader = DataLoader(
            selection_dataset, batch_size=args.batch_size, shuffle=True
        )
    action_dim = actions.shape[1]
    model_class = ColorAttentionVLA if args.model_type == "color_attention" else MiniVLA
    model = model_class(
        len(vocabulary), states.shape[1], args.chunk_size * action_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    weights = torch.tensor(
        [2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 1.0], device=device
    ).repeat(args.chunk_size)
    best = float("inf")

    print(
        f"device={device}, train={len(datasets[0])}, validation={len(datasets[1])}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, args.epochs + 1):
        totals = []
        model.train()
        train_total = 0.0
        for image, tokens, state, target in train_loader:
            image, tokens = image.to(device), tokens.to(device)
            state, target = state.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(image, tokens, state) - target).square() * weights).mean()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * image.shape[0]
        if selection_loader is not None:
            for image, tokens, state, target in selection_loader:
                image, tokens = image.to(device), tokens.to(device)
                state, target = state.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                first_action = model(image, tokens, state)[:, :action_dim]
                selection_loss = (
                    (first_action - target).square() * weights[:action_dim]
                ).mean()
                (args.selection_loss_weight * selection_loss).backward()
                optimizer.step()
        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for image, tokens, state, target in validation_loader:
                image, tokens = image.to(device), tokens.to(device)
                state, target = state.to(device), target.to(device)
                loss = ((model(image, tokens, state) - target).square() * weights).mean()
                validation_total += loss.item() * image.shape[0]
        train_loss = train_total / len(datasets[0])
        validation_loss = validation_total / len(datasets[1])
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )
        if validation_loss < best:
            best = validation_loss
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
                    "model_type": "color_attention" if args.model_type == "color_attention" else "mini",
                },
                args.output,
            )
    print(f"best validation loss={best:.6f}")
    print(f"checkpoint saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
