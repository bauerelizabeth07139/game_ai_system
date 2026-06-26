"""噪声注入器 - 模拟人类操作抖动"""
import logging
import math
import time
import numpy as np

logger = logging.getLogger(__name__)


class NoiseInjector:
    """垂直噪声注入器

    频率8Hz振幅0.02的正弦波模拟抖动
    """

    def __init__(self, config: dict = None):
        self.frequency = 8.0  # Hz
        self.amplitude = 0.02

        if config:
            self.frequency = config.get('frequency', 8.0)
            self.amplitude = config.get('amplitude', 0.02)

        self._start_time = time.perf_counter()

    def get_noise(self) -> float:
        """获取当前时刻的噪声值

        Returns:
            float: 噪声值，范围约 [-amplitude, amplitude]
        """
        elapsed = time.perf_counter() - self._start_time
        noise = self.amplitude * math.sin(2 * math.pi * self.frequency * elapsed)
        return noise

    def apply_to_mouse(self, dx: float, dy: float) -> tuple:
        """对鼠标偏移应用噪声

        Args:
            dx: 水平偏移
            dy: 垂直偏移

        Returns:
            tuple: (dx, dy+noise)
        """
        noise = self.get_noise()
        return (dx, dy + noise)

    def reset(self):
        """重置起始时间"""
        self._start_time = time.perf_counter()
