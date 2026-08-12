"""Compound language planner with visual scene memory and sequential execution."""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from stage7_skill import execute_visual_place
from stage7_vision import BallVisionTracker

COLORS_ZH = {"红": "red", "红色": "red", "绿": "green", "绿色": "green"}


@dataclass(frozen=True)
class PlanStep:
    target: str
    relation: str
    reference: str
    source_text: str


@dataclass
class StepResult:
    step: PlanStep
    goal: np.ndarray
    completed: bool
    success: bool
    phase: str
    control_steps: int
    attempts: int = 1


def parse_compound_instruction(text: str) -> list[PlanStep]:
    normalized = text.strip().lower().replace("的球", "球")
    matches = list(re.finditer(
        r"把\s*(红色|绿色|红|绿)球?\s*放到\s*(红色|绿色|红|绿)"
        r"(?:球)?\s*(旁边|正前方|前方|正前面|前面|原来的位置|原位置)", normalized
    ))
    steps = []
    for match in matches:
        relation = {
            "旁边": "beside",
            "正前方": "front",
            "前方": "front",
            "正前面": "front",
            "前面": "front",
            "原来的位置": "original",
            "原位置": "original",
        }[match.group(3)]
        steps.append(PlanStep(
            COLORS_ZH[match.group(1)], relation, COLORS_ZH[match.group(2)],
            match.group(0),
        ))
    if not steps:
        english = re.finditer(
            r"(?:place|put|move)\s+(?:the\s+)?(red|green)\s+(?:ball\s+)?"
            r"(next to|beside|in front of|at)\s+(?:the\s+)?(red|green)(?:'s)?\s*"
            r"(original (?:position|location))?", normalized
        )
        for match in english:
            relation = (
                "original" if match.group(4)
                else "front" if match.group(2) == "in front of"
                else "beside"
            )
            steps.append(PlanStep(
                match.group(1), relation, match.group(3), match.group(0),
            ))
    if not steps:
        raise ValueError("无法解析任务。请使用“把绿色的球放到红色旁边”这样的表达。")
    expected = len(re.findall(
        r"把\s*(?:红色|绿色|红|绿).*?放到", normalized
    ))
    expected += len(re.findall(r"\b(?:place|put|move)\b", normalized))
    if expected != len(steps):
        raise ValueError(
            f"指令包含 {expected} 个动作，但只识别出 {len(steps)} 个；"
            "系统已停止，避免只执行部分任务。"
        )
    for step in steps:
        if step.target == step.reference:
            raise ValueError("目标球和参考球不能是同一个颜色。")
    return steps


def capture_scene_memory(env, observation, image_mode="original"):
    tracker = BallVisionTracker(env.sim, smoothing=1.0)
    detections = tracker.update(observation, image_mode)
    if any(detections[color] is None for color in ("red", "green")):
        raise RuntimeError("vision_failed: 初始画面没有同时检测到红球和绿球")
    return {color: tracker.require(color) for color in ("red", "green")}


def safe_beside_goal(reference, future_destination=None):
    """Place beside the reference but outside the following transport corridor."""
    if future_destination is None:
        direction = -1.0 if reference[0] >= 0 else 1.0
        return np.array([reference[0] + direction * 0.15, reference[1], 0.818])
    travel = np.asarray(future_destination)[:2] - np.asarray(reference)[:2]
    norm = np.linalg.norm(travel)
    if norm < 1e-6:
        perpendicular = np.array([1.0, 0.0])
    else:
        perpendicular = np.array([-travel[1], travel[0]]) / norm
    candidates = [
        np.asarray(reference)[:2] + sign * 0.15 * perpendicular
        for sign in (-1.0, 1.0)
    ]
    in_view = [p for p in candidates if abs(p[0]) < 0.22 and abs(p[1]) < 0.22]
    choices = in_view or candidates
    # Prefer the far side of the table, away from the robot's approach corridor.
    xy = max(choices, key=lambda point: point[1])
    return np.array([xy[0], xy[1], 0.818])


def front_goal(reference, camera_xy):
    """Camera-relative front: move toward the viewer on the table plane."""
    reference_xy = np.asarray(reference)[:2]
    direction = np.asarray(camera_xy)[:2] - reference_xy
    norm = max(float(np.linalg.norm(direction)), 1e-6)
    goal_xy = reference_xy + 0.12 * direction / norm
    return np.array([goal_xy[0], goal_xy[1], 0.818])


def move_to_observation_pose(env, observation, frame_callback=None):
    """Clear the workspace and let objects settle between sub-tasks."""
    waypoint = np.array([0.0, -0.20, 1.05])
    for _ in range(70):
        error = waypoint - observation["robot0_eef_pos"]
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[:3] = np.clip(error * 10.0, -0.65, 0.65)
        action[-1] = -1.0
        observation, _, _, _ = env.step(action)
        if frame_callback:
            frame_callback(observation)
        if np.linalg.norm(error) < 0.012:
            break
    for _ in range(12):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = -1.0
        observation, _, _, _ = env.step(action)
        if frame_callback:
            frame_callback(observation)
    return observation


def visual_goal_error(env, observation, color, goal, image_mode):
    tracker = BallVisionTracker(env.sim, smoothing=1.0)
    detections = tracker.update(observation, image_mode)
    if detections[color] is None:
        return float("inf")
    return float(np.linalg.norm(tracker.require(color)[:2] - goal[:2]))


def execute_plan(env, observation, instruction, max_steps=220, render=False,
                 image_mode="original", progress=None, frame_callback=None):
    plan = parse_compound_instruction(instruction)
    memory = capture_scene_memory(env, observation, image_mode)
    results = []
    for index, step in enumerate(plan, 1):
        tracker = BallVisionTracker(env.sim, smoothing=1.0)
        detections = tracker.update(observation, image_mode)
        if any(detections[color] is None for color in ("red", "green")):
            results.append(StepResult(step, np.zeros(3), False, False,
                                      "vision_failed", 0))
            break
        reference = tracker.require(step.reference)
        future_destination = None
        if index < len(plan):
            next_step = plan[index]
            if next_step.relation == "original":
                future_destination = memory[next_step.reference]
        if step.relation == "beside":
            goal = safe_beside_goal(reference, future_destination)
        elif step.relation == "front":
            camera_id = env.sim.model.camera_name2id("agentview")
            goal = front_goal(reference, env.sim.data.cam_xpos[camera_id, :2])
        else:
            goal = memory[step.reference].copy()
        goal[2] = 0.818
        if progress:
            progress(f"执行 {index}/{len(plan)}：{step.source_text}")
        observation, _, success, control_steps, phase = execute_visual_place(
            env, observation, step.target, goal, max_steps, render, image_mode,
            frame_callback, motion_limit=1.0, track_until_descend=index > 1,
            release_height=0.004, settle_steps=50,
            open_action=-1.0, open_steps=18, transport_height=0.24,
        )
        completed = phase == "done"
        results.append(StepResult(step, goal, completed, bool(success),
                                  phase, control_steps))
        if not completed:
            break
        if index < len(plan):
            if progress:
                progress("返回安全观察位，等待物体稳定")
            observation = move_to_observation_pose(
                env, observation, frame_callback
            )
    # Task-level visual verification: later motions can disturb an earlier ball.
    # Retry only the semantic goals that are no longer satisfied.
    if len(results) == len(plan) and all(result.completed for result in results):
        for repair_round in range(2):
            pending = [
                result for result in results
                if visual_goal_error(
                    env, observation, result.step.target, result.goal, image_mode
                ) > 0.03
            ]
            if not pending:
                break
            for result in pending:
                if progress:
                    progress(
                        f"视觉复核第 {repair_round + 1} 轮：修正 {result.step.target}"
                    )
                observation = move_to_observation_pose(
                    env, observation, frame_callback
                )
                observation, _, success, used_steps, phase = execute_visual_place(
                    env, observation, result.step.target, result.goal,
                    max_steps, render, image_mode, frame_callback,
                    motion_limit=1.0, track_until_descend=True,
                    release_height=0.004, settle_steps=50,
                    open_action=-1.0, open_steps=18, transport_height=0.24,
                )
                result.control_steps += used_steps
                result.attempts += 1
                result.phase = phase
                result.completed = phase == "done"
                result.success = bool(success)
                if not result.completed:
                    break
        for result in results:
            result.success = (
                result.completed
                and visual_goal_error(
                    env, observation, result.step.target, result.goal, image_mode
                ) <= 0.03
            )
    return observation, plan, memory, results
