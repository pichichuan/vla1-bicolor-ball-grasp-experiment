"""Tk GUI for entering and executing compound visual manipulation commands."""
from __future__ import annotations
import argparse
import multiprocessing as mp
import queue
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import stage6_env  # noqa: F401
from stage6_skill import reset_nontrivial
from stage8_planner import execute_plan, parse_compound_instruction
from vla_common import make_env

DEFAULT_TEXT = "把绿色的球放到红色旁边。再把红色的球放到绿色原来的位置"
DISPLAY_RENDER_SIZE = 480  # High-resolution GUI preview; control RGB remains 96x96.
DISPLAY_SIZE = 600


class SimulationWorker(mp.Process):
    def __init__(self, commands, events):
        super().__init__(daemon=True)
        self.commands, self.events = commands, events
        self.observation = None
        self.frame_count = 0

    def emit_frame(self, observation, force=False):
        self.frame_count += 1
        if not force and self.frame_count % 4:
            return
        # Keep the VLA control observation at 96x96, but render a separate
        # high-resolution frame for people watching the desktop GUI.
        image = np.ascontiguousarray(self.env.sim.render(
            width=DISPLAY_RENDER_SIZE,
            height=DISPLAY_RENDER_SIZE,
            camera_name="agentview",
        )[::-1])
        self.events.put((
            "frame", (image.shape[1], image.shape[0], image.tobytes())
        ))

    @staticmethod
    def diagnostic(text):
        path = Path(__file__).resolve().parent / "reports" / "stage8_gui_worker.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()

    def run(self):
        env = None
        try:
            self.diagnostic("worker_started")
            self.diagnostic("creating_environment")
            env = make_env(renderer=False, seed=8901, env_name="ColorBallPlace")
            self.env = env
            self.diagnostic("environment_created")
            self.observation = reset_nontrivial(env)
            self.diagnostic("first_reset_complete")
            self.emit_frame(self.observation, force=True)
            self.diagnostic("first_frame_sent")
            self.events.put(("ready", "环境已就绪，请输入任务。"))
            self.diagnostic("ready_sent")
            while True:
                command, payload = self.commands.get()
                if command == "close":
                    break
                if command == "reset":
                    self.observation = reset_nontrivial(env)
                    self.emit_frame(self.observation, force=True)
                    self.events.put(("ready", "场景已重置。"))
                elif command == "run":
                    try:
                        self.events.put(("log", f"收到指令：{payload}"))
                        final, plan, _, results = execute_plan(
                            env, self.observation, payload,
                            progress=lambda text: self.events.put(("log", text)),
                            frame_callback=self.emit_frame,
                        )
                        self.observation = final
                        completed = len(results) == len(plan) and all(r.completed for r in results)
                        self.events.put(("ready", f"执行结束：{len(results)}/{len(plan)} 步，完成={completed}"))
                    except Exception as error:
                        self.events.put(("error", str(error)))
        except Exception as error:
            self.diagnostic("worker_error\n" + traceback.format_exc())
            self.events.put(("error", f"MuJoCo 后台启动或执行失败：{error!r}"))
        finally:
            if env is not None:
                env.close()


class Stage8GUI:
    def __init__(self, root):
        self.root = root
        root.title("MuJoCo 视觉语言机械臂 · 第八阶段")
        root.geometry("820x900")
        root.minsize(760, 820)
        self.commands, self.events = mp.Queue(), mp.Queue()
        self.worker = SimulationWorker(self.commands, self.events)

        ttk.Label(root, text="视觉语言多任务控制", font=("Microsoft YaHei UI", 18, "bold")).pack(pady=(16, 6))
        ttk.Label(root, text="输入复合指令；系统将解析、记忆初始位置并顺序执行。后端只用 RGB 定位。",
                  font=("Microsoft YaHei UI", 10)).pack()
        self.image_label = ttk.Label(root, text="正在启动 MuJoCo…", anchor="center")
        self.image_label.pack(pady=10)

        input_frame = ttk.Frame(root)
        input_frame.pack(fill="x", padx=24)
        self.entry = ttk.Entry(input_frame, font=("Microsoft YaHei UI", 11))
        self.entry.insert(0, DEFAULT_TEXT)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.run_button = ttk.Button(input_frame, text="执行", command=self.run_task, state="disabled")
        self.run_button.pack(side="left")
        self.reset_button = ttk.Button(input_frame, text="重置场景", command=self.reset, state="disabled")
        self.reset_button.pack(side="left", padx=(8, 0))

        self.status = tk.StringVar(value="正在加载仿真环境…")
        ttk.Label(root, textvariable=self.status).pack(anchor="w", padx=24, pady=(12, 4))
        self.log = tk.Text(root, height=7, state="disabled", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker.start()
        root.after(50, self.poll)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_busy(self, busy, text):
        state = "disabled" if busy else "normal"
        self.run_button.configure(state=state)
        self.reset_button.configure(state=state)
        self.status.set(text)

    def run_task(self):
        instruction = self.entry.get().strip()
        try:
            plan = parse_compound_instruction(instruction)
        except ValueError as error:
            messagebox.showerror("无法解析", str(error))
            return
        self.append_log(f"规划得到 {len(plan)} 个步骤。")
        self.set_busy(True, "机械臂正在执行…")
        self.commands.put(("run", instruction))

    def reset(self):
        self.set_busy(True, "正在重置场景…")
        self.commands.put(("reset", None))

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    try:
                        width, height, pixels = payload
                        pil_image = Image.frombytes(
                            "RGB", (width, height), pixels
                        ).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS)
                        image = ImageTk.PhotoImage(pil_image)
                        self.image_label.configure(image=image, text="")
                        self.image_label.image = image
                    except tk.TclError as error:
                        self.append_log(f"相机画面显示失败：{error}")
                elif kind == "log":
                    self.append_log(payload)
                elif kind == "ready":
                    self.append_log(payload)
                    self.set_busy(False, payload)
                elif kind == "error":
                    self.append_log("错误：" + payload)
                    self.set_busy(False, "执行失败")
                    messagebox.showerror("执行失败", payload)
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def close(self):
        self.commands.put(("close", None))
        self.worker.join(timeout=1.0)
        if self.worker.is_alive():
            self.worker.terminate()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        interpreter = tk.Tcl()
        plan = parse_compound_instruction(DEFAULT_TEXT)
        print(f"tk_version={interpreter.eval('info patchlevel')}")
        print(f"planned_steps={len(plan)}")
        print([(step.target, step.relation, step.reference) for step in plan])
        return
    root = tk.Tk()
    Stage8GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
