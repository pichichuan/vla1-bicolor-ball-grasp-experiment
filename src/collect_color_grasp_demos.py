"""Collect paired red-vs-green grasp demonstrations from identical scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grasp_teacher import GraspTeacher
from vla_common import get_image, get_state, make_env


COLOR_TASKS = {
    "red": {
        "key": "cubeA_pos",
        "other_key": "cubeB_pos",
        "phrases": ("pick up the red cube", "grasp the red block"),
    },
    "green": {
        "key": "cubeB_pos",
        "other_key": "cubeA_pos",
        "phrases": ("pick up the green cube", "grasp the green block"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/color_grasp_demos.npz"))
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def collect_episode(env, observation, object_key: str, max_steps: int):
    teacher = GraspTeacher(env.action_dim, object_key=object_key)
    samples = []
    for _ in range(max_steps):
        action, status = teacher.act(observation)
        samples.append((get_image(observation), get_state(observation), action.copy()))
        observation, _, _, _ = env.step(action)
        if status.phase == "done":
            break
    success = float(observation[object_key][2]) > 0.84
    return samples, observation, success


def main() -> None:
    args = parse_args()
    # Independent environments with the same RNG seed produce paired, identical
    # initial scenes while allowing the red and green trajectories to diverge.
    environments = {
        color: make_env(renderer=False, seed=args.seed, env_name="Stack")
        for color in COLOR_TASKS
    }
    images, states, actions = [], [], []
    instructions, colors = [], []
    episode_ids, scene_ids, step_ids = [], [], []
    initial_positions = []

    try:
        scene_id = 0
        attempt = 0
        while scene_id < args.scenes and attempt < args.max_attempts:
            attempt += 1
            observations = {color: env.reset() for color, env in environments.items()}
            paired_state = environments["red"].sim.get_state().flatten()
            environments["green"].sim.set_state_from_flattened(paired_state)
            environments["green"].sim.forward()
            observations["green"] = environments["green"]._get_observations(
                force_update=True
            )
            red_layout = np.concatenate(
                [observations["red"]["cubeA_pos"], observations["red"]["cubeB_pos"]]
            )
            green_layout = np.concatenate(
                [observations["green"]["cubeA_pos"], observations["green"]["cubeB_pos"]]
            )
            if not np.allclose(red_layout, green_layout, atol=1e-6):
                raise RuntimeError("Paired environments produced different initial scenes")
            pair_results = []
            pair_valid = True
            for color_index, (color, task) in enumerate(COLOR_TASKS.items()):
                phrase = task["phrases"][scene_id % len(task["phrases"])]
                samples, final_observation, success = collect_episode(
                    environments[color],
                    observations[color],
                    task["key"],
                    args.max_steps,
                )
                wrong_lifted = float(final_observation[task["other_key"]][2]) > 0.84
                if not success or wrong_lifted:
                    pair_valid = False
                pair_results.append((color_index, color, phrase, samples))

            if not pair_valid:
                print(f"attempt={attempt:03d} discarded paired scene")
                continue

            initial_positions.append(red_layout)
            for color_index, color, phrase, samples in pair_results:
                episode_id = scene_id * 2 + color_index
                for step_id, (image, state, action) in enumerate(samples):
                    images.append(image)
                    states.append(state)
                    actions.append(action)
                    instructions.append(phrase)
                    colors.append(color)
                    episode_ids.append(episode_id)
                    scene_ids.append(scene_id)
                    step_ids.append(step_id)
            print(
                f"scene={scene_id + 1:03d}/{args.scenes:03d} "
                f"red_xy={red_layout[:2]} green_xy={red_layout[3:5]}"
            )
            scene_id += 1
    finally:
        for env in environments.values():
            env.close()

    if scene_id < args.scenes:
        raise RuntimeError(
            f"Only collected {scene_id}/{args.scenes} valid paired scenes "
            f"within {args.max_attempts} attempts"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        images=np.stack(images),
        states=np.stack(states),
        actions=np.stack(actions),
        instructions=np.asarray(instructions),
        colors=np.asarray(colors),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        scene_ids=np.asarray(scene_ids, dtype=np.int32),
        step_ids=np.asarray(step_ids, dtype=np.int32),
        initial_positions=np.stack(initial_positions),
    )
    print(
        f"saved {len(images)} frames from {args.scenes} paired scenes "
        f"({args.scenes * 2} episodes) to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
