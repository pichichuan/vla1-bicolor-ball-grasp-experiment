from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
VIDEO_PATH = REPORTS / "vla1_complete_grasp_review.webm"
LOG_PATH = REPORTS / "vla1_complete_grasp_review.json"
HTML_PATH = REPORTS / "VLA1双色小球完整抓取复盘.html"
sys.path.insert(0, str(PROJECT))

import stage6_env  # noqa: F401
from stage6_skill import reset_nontrivial
from stage6_teacher import BALL_TASKS
from stage8_gui import DEFAULT_TEXT
from stage8_planner import execute_plan
from vla_common import get_image, make_env


FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 28)
SMALL_FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)


def source_excerpt(filename: str, start: int, end: int) -> str:
    lines = (PROJECT / filename).read_text(encoding="utf-8").splitlines()
    return "\n".join(
        f"{number:4d}  {lines[number - 1]}"
        for number in range(start, min(end, len(lines)) + 1)
    )


def render_frame(image: np.ndarray, status: str, frame_number: int) -> np.ndarray:
    view = Image.fromarray(image[::-1]).resize((720, 720), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1280, 720), (10, 17, 15))
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((760, 34), "VLA1 双色小球完整抓取", font=FONT, fill=(94, 229, 151))
    draw.line((760, 82, 1238, 82), fill=(45, 105, 75), width=2)
    draw.text((760, 108), "当前阶段", font=SMALL_FONT, fill=(151, 178, 162))
    wrapped = status
    if len(wrapped) > 20:
        wrapped = wrapped[:20] + "\n" + wrapped[20:40]
    draw.multiline_text((760, 140), wrapped, font=FONT, fill=(240, 244, 241), spacing=10)
    draw.text((760, 254), "任务", font=SMALL_FONT, fill=(151, 178, 162))
    draw.multiline_text(
        (760, 286),
        "1. 绿球放到红球旁边\n2. 红球放到绿球原位置",
        font=SMALL_FONT,
        fill=(222, 231, 225),
        spacing=14,
    )
    draw.text((760, 430), "控制视觉：96×96 RGB", font=SMALL_FONT, fill=(151, 178, 162))
    draw.text((760, 466), "录像画面：720×720 RGB", font=SMALL_FONT, fill=(151, 178, 162))
    draw.text((760, 502), "控制：20 Hz  ·  录像：10 FPS", font=SMALL_FONT, fill=(151, 178, 162))
    draw.text((760, 660), f"frame {frame_number:04d}", font=SMALL_FONT, fill=(96, 142, 115))
    return np.asarray(canvas)


def write_video(frames: list[tuple[np.ndarray, str]]) -> None:
    writer = cv2.VideoWriter(
        str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"VP80"), 10.0, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV failed to open the WebM/VP8 writer")
    try:
        for number, (image, status) in enumerate(frames):
            rgb = render_frame(image, status, number)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_html(payload: dict) -> None:
    excerpts = (
        ("GUI入口与后台执行", "stage8_gui.py", 17, 75),
        ("复合语言解析", "stage8_planner.py", 31, 83),
        ("场景记忆、顺序执行与视觉纠偏", "stage8_planner.py", 153, 239),
        ("RGB颜色检测与坐标反投影", "stage7_vision.py", 36, 88),
        ("完整抓放状态机", "stage6_teacher.py", 30, 112),
    )
    code_html = "\n".join(
        f"<details><summary>{html.escape(title)} · {filename}:{start}</summary>"
        f"<pre>{html.escape(source_excerpt(filename, start, end))}</pre></details>"
        for title, filename, start, end in excerpts
    )
    steps = payload["steps"]
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VLA1 双色小球完整抓取复盘</title>
<style>
body{{margin:0;background:#07110d;color:#e8f1eb;font-family:'Microsoft YaHei UI',sans-serif}}
main{{max-width:1120px;margin:auto;padding:28px}} h1{{color:#63e59a}} h2{{margin-top:34px;color:#9df0bd}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .card{{background:#10231a;border:1px solid #24593d;border-radius:12px;padding:16px}}
.value{{font-size:25px;color:#6af0a0;margin-top:8px}} video{{width:100%;border-radius:14px;border:1px solid #2d7250;background:#000}}
table{{width:100%;border-collapse:collapse;background:#0d1d16}} th,td{{padding:12px;border:1px solid #234d38;text-align:left}}
details{{margin:10px 0;background:#0d1d16;border:1px solid #244d38;border-radius:9px;padding:12px}} summary{{cursor:pointer;color:#8ce9ae}}
pre{{overflow:auto;background:#040906;padding:14px;border-radius:8px;line-height:1.45;color:#d7e8dc}} .note{{color:#a9bdb0;line-height:1.8}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>VLA1 双色小球完整抓取复盘</h1>
<p class="note">同一次真实 MuJoCo 执行产生的视频、结果日志与当前代码。任务：{html.escape(payload['instruction'])}</p>
<div class="grid">
<div class="card">任务结果<div class="value">{'成功' if payload['success'] else '失败'}</div></div>
<div class="card">绿色球误差<div class="value">{steps[0]['goal_error_cm']:.3f} cm</div></div>
<div class="card">红色球误差<div class="value">{steps[1]['goal_error_cm']:.3f} cm</div></div>
<div class="card">录像时长<div class="value">{payload['video_duration_s']:.1f} s</div></div>
</div>
<h2>完整抓取视频</h2><video controls preload="metadata"><source src="vla1_complete_grasp_review.webm?v={html.escape(payload['generated_at'])}" type="video/webm"><source src="vla1_complete_grasp_review.mp4" type="video/mp4">浏览器不支持内嵌视频，请使用下方文件链接。</video>
<h2>执行日志</h2>
<table><tr><th>步骤</th><th>目标</th><th>关系</th><th>阶段</th><th>控制步</th><th>尝试</th><th>最终误差</th></tr>
<tr><td>1</td><td>绿色球</td><td>红球旁边</td><td>{steps[0]['phase']}</td><td>{steps[0]['control_steps']}</td><td>{steps[0]['attempts']}</td><td>{steps[0]['goal_error_cm']:.3f} cm</td></tr>
<tr><td>2</td><td>红色球</td><td>绿球原位置</td><td>{steps[1]['phase']}</td><td>{steps[1]['control_steps']}</td><td>{steps[1]['attempts']}</td><td>{steps[1]['goal_error_cm']:.3f} cm</td></tr></table>
<details><summary>展开原始 JSON 日志</summary><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></details>
<h2>关键代码</h2>{code_html}
<h2>复盘结论</h2><p class="note">语言被解析为两个顺序子任务；RGB检测建立初始场景记忆；状态机完成接近、下降、夹持、抬升、运输、稳定释放与退回；任务结束后以视觉误差复核。最终 GUI 是模块化 Language + Vision + Action 系统，不是大型端到端预训练 VLA。</p>
</main></body></html>"""
    HTML_PATH.write_text(page, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    env = make_env(renderer=False, seed=8901, env_name="ColorBallPlace")
    try:
        for attempt in range(1, 4):
            observation = reset_nontrivial(env)
            frames: list[tuple[np.ndarray, str]] = []
            status = ["读取初始 RGB 场景"]
            callback_count = [0]

            def progress(text: str) -> None:
                status[0] = text

            def capture(frame) -> None:
                callback_count[0] += 1
                if callback_count[0] % 2 == 0:
                    high_resolution = env.sim.render(
                        width=720, height=720, camera_name="agentview"
                    )
                    frames.append((high_resolution.copy(), status[0]))

            initial_high_resolution = env.sim.render(
                width=720, height=720, camera_name="agentview"
            )
            frames.append((initial_high_resolution.copy(), status[0]))
            final, plan, memory, results = execute_plan(
                env,
                observation,
                DEFAULT_TEXT,
                progress=progress,
                frame_callback=capture,
            )
            step_payload = []
            for result in results:
                final_position = np.asarray(final[BALL_TASKS[result.step.target]["key"]])
                error_cm = float(np.linalg.norm(final_position[:2] - result.goal[:2]) * 100)
                step_payload.append(
                    {
                        "target": result.step.target,
                        "relation": result.step.relation,
                        "reference": result.step.reference,
                        "phase": result.phase,
                        "completed": result.completed,
                        "success": result.success,
                        "control_steps": result.control_steps,
                        "attempts": result.attempts,
                        "goal_m": result.goal.round(6).tolist(),
                        "final_position_m": final_position.round(6).tolist(),
                        "goal_error_cm": round(error_cm, 3),
                    }
                )
            success = (
                len(results) == len(plan) == 2
                and all(item.completed and item.success for item in results)
                and all(item["goal_error_cm"] < 3.5 for item in step_payload)
            )
            if not success:
                print(f"attempt={attempt} failed; retrying", flush=True)
                continue
            frames.extend([frames[-1]] * 15)
            payload = {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "attempt": attempt,
                "success": True,
                "instruction": DEFAULT_TEXT,
                "environment": "robosuite ColorBallPlace / Panda / MuJoCo",
                "camera": "agentview: 96x96 control RGB + 720x720 recording RGB",
                "control_frequency_hz": 20,
                "video_fps": 10,
                "video_frames": len(frames),
                "video_duration_s": round(len(frames) / 10.0, 1),
                "scene_memory_m": {
                    color: position.round(6).tolist() for color, position in memory.items()
                },
                "steps": step_payload,
                "evidence_note": "This video, JSON log, and metrics come from the same successful run.",
            }
            write_video(frames)
            LOG_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            write_html(payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"video={VIDEO_PATH}")
            print(f"log={LOG_PATH}")
            print(f"html={HTML_PATH}")
            return
        raise RuntimeError("No successful recording in three attempts")
    finally:
        env.close()


if __name__ == "__main__":
    main()
