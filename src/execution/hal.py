"""HAL硬件抽象层 - 40维动作输出"""
import logging
import time
import threading
import queue
import numpy as np
from .pid_controller import PIDController
from .noise_injector import NoiseInjector

logger = logging.getLogger(__name__)

# 40维动作名称
ACTION_NAMES = [
    'Mx', 'My', 'Wx', 'Wy',   # 0-3: 连续控制
    'LMB', 'RMB', 'MMB',       # 4-6: 鼠标
    'W', 'A', 'S', 'D',        # 7-10: 移动
    'Space', 'Shift', 'Ctrl', 'Tab', 'Esc',  # 11-15: 功能
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',  # 16-25: 数字
    'Q', 'E', 'R', 'F', 'G', 'H', 'Z', 'X', 'C', 'V',  # 26-35: 技能
    'B', 'N', 'T', 'M'         # 36-39: 其他
]

# 按键索引到pynput Key名称的映射
KEY_MAP = {
    7: 'w', 8: 'a', 9: 's', 10: 'd',
    11: 'space', 12: 'shift', 13: 'ctrl', 14: 'tab', 15: 'esc',
    16: '1', 17: '2', 18: '3', 19: '4', 20: '5',
    21: '6', 22: '7', 23: '8', 24: '9', 25: '0',
    26: 'q', 27: 'e', 28: 'r', 29: 'f', 30: 'g',
    31: 'h', 32: 'z', 33: 'x', 34: 'c', 35: 'v',
    36: 'b', 37: 'n', 38: 't', 39: 'm'
}


class HAL:
    """硬件抽象层 - 40维动作"""

    def __init__(self, config: dict = None, real_input: bool = False):
        self.real_input = real_input
        self._mouse_controller = None
        self._keyboard_controller = None

        pid_config = config.get('pid', {}) if config else {}
        noise_config = config.get('noise', {}) if config else {}
        self.mouse_sensitivity = (config.get('mouse_sensitivity', 100) if config else 100)
        self.frame_interval = 0.1

        self.pid = PIDController(pid_config)
        self.noise = NoiseInjector(noise_config)

        self._action_queue = queue.Queue(maxsize=10)
        self._running = False
        self._executor_thread = None

        if self.real_input:
            self._init_pynput()

    def _init_pynput(self):
        try:
            from pynput.mouse import Button, Controller as MouseController
            from pynput.keyboard import Controller as KeyboardController, Key
            self._mouse_controller = MouseController()
            self._keyboard_controller = KeyboardController()
            self._Button = Button
            self._Key = Key
            logger.info("pynput初始化成功")
        except Exception as e:
            logger.warning(f"pynput初始化失败: {e}")
            self.real_input = False

    def start(self):
        self._running = True
        self._executor_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._executor_thread.start()
        logger.info(f"执行层已启动 (真实输入: {self.real_input})")

    def stop(self):
        self._running = False
        if self._executor_thread:
            self._executor_thread.join(timeout=2)
        logger.info("执行层已停止")

    def execute(self, action_vector: np.ndarray):
        """执行40维动作向量"""
        try:
            self._action_queue.put_nowait(action_vector)
        except queue.Full:
            pass

    def _execution_loop(self):
        while self._running:
            try:
                action = self._action_queue.get(timeout=self.frame_interval)
                self._process_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"执行动作异常: {e}")

    def _process_action(self, action: np.ndarray):
        """处理40维动作"""
        # 连续控制：Sigmoid[0,1] -> 映射到[-1,1]
        mouse_vx = (action[0] - 0.5) * 2.0
        mouse_vy = (action[1] - 0.5) * 2.0
        move_vx = (action[2] - 0.5) * 2.0
        move_vy = (action[3] - 0.5) * 2.0

        # PID平滑鼠标速度
        smooth_v = self.pid.update(np.array([mouse_vx, mouse_vy]))
        velocity_x, velocity_y = self.noise.apply_to_mouse(float(smooth_v[0]), float(smooth_v[1]))

        # 速度转位移
        mouse_dx = velocity_x * self.mouse_sensitivity * self.frame_interval
        mouse_dy = velocity_y * self.mouse_sensitivity * self.frame_interval

        # 按键概率（阈值0.5）
        button_triggers = action[4:40] > 0.5

        # 日志：只显示激活的按键
        active_keys = [ACTION_NAMES[i + 4] for i, t in enumerate(button_triggers) if t]
        keys_str = ",".join(active_keys[:8]) if active_keys else "-"
        if len(active_keys) > 8:
            keys_str += f"...+{len(active_keys)-8}"

        logger.info(f"鼠标:({velocity_x:+.2f},{velocity_y:+.2f}) 移动:({move_vx:+.2f},{move_vy:+.2f}) [{keys_str}]")

        if self.real_input:
            self._apply_real_input(mouse_dx, mouse_dy, move_vx, move_vy, button_triggers)

    def _apply_real_input(self, mouse_dx, mouse_dy, move_vx, move_vy, buttons):
        if not self._mouse_controller or not self._keyboard_controller:
            return
        try:
            dx, dy = int(mouse_dx), int(mouse_dy)
            if dx != 0 or dy != 0:
                self._mouse_controller.move(dx, dy)

            if buttons[0]:  # LMB
                self._mouse_controller.click(self._Button.left)
            if buttons[1]:  # RMB
                self._mouse_controller.click(self._Button.right)
            if buttons[2]:  # MMB
                self._mouse_controller.click(self._Button.middle)

            # 键盘按键
            for idx, key_name in KEY_MAP.items():
                btn_idx = idx - 4
                if btn_idx < len(buttons) and buttons[btn_idx]:
                    try:
                        key = getattr(self._Key, key_name, key_name)
                        self._keyboard_controller.press(key)
                        self._keyboard_controller.release(key)
                    except Exception:
                        pass

            # 滚轮（通过ScrU/ScrD检测）
            # 这里没有绑定滚轮到40维，暂时跳过
        except Exception as e:
            logger.error(f"真实输入失败: {e}")

    def format_action_summary(self, action: np.ndarray) -> str:
        """格式化动作摘要"""
        if len(action) < 40:
            return "无效动作向量"

        mouse_vx = (action[0] - 0.5) * 2.0
        mouse_vy = (action[1] - 0.5) * 2.0
        move_vx = (action[2] - 0.5) * 2.0
        move_vy = (action[3] - 0.5) * 2.0
        buttons = action[4:40]

        active_keys = [ACTION_NAMES[i + 4] for i, p in enumerate(buttons) if p > 0.5]
        keys_str = ",".join(active_keys[:6]) if active_keys else "-"
        if len(active_keys) > 6:
            keys_str += f"+{len(active_keys)-6}"

        return (
            f"鼠标({mouse_vx:+.2f},{mouse_vy:+.2f}) "
            f"移动({move_vx:+.2f},{move_vy:+.2f}) "
            f"[{keys_str}]"
        )
