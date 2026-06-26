"""PID控制器 - 鼠标移动速度平滑处理"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PIDController:
    """PID控制器，用于鼠标移动速度平滑
    
    控制目标：速度（velocity），而非位移（displacement）
    - 输入：目标速度 [-1, 1]
    - 输出：平滑后的速度 [-1, 1]
    
    参数：
        Kp: 比例增益 (默认1.2)
        Ki: 积分增益 (默认0.01)
        Kd: 微分增益 (默认0.1)
    """

    def __init__(self, config: dict = None):
        self.Kp = 1.2
        self.Ki = 0.01
        self.Kd = 0.1

        if config:
            self.Kp = config.get('Kp', self.Kp)
            self.Ki = config.get('Ki', self.Ki)
            self.Kd = config.get('Kd', self.Kd)

        # 内部状态
        self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._integral = np.zeros(2, dtype=np.float64)
        self._initialized = False

    def reset(self):
        """重置PID状态"""
        self._prev_velocity = np.zeros(2, dtype=np.float64)
        self._integral = np.zeros(2, dtype=np.float64)
        self._initialized = False

    def update(self, target_velocity: np.ndarray) -> np.ndarray:
        """计算平滑后的速度

        Args:
            target_velocity: 目标速度 [vx, vy]，范围[-1, 1]

        Returns:
            np.ndarray: 平滑后的速度 [vx, vy]，范围[-1, 1]
        """
        target = np.array(target_velocity, dtype=np.float64)

        if not self._initialized:
            self._prev_velocity = target
            self._initialized = True
            return target.astype(np.float32)

        # 速度误差：目标速度与当前速度之差
        error = target - self._prev_velocity
        
        # 积分项（累积速度误差）
        self._integral += error
        
        # 微分项（速度变化率）
        derivative = error
        
        # PID输出：速度修正量
        correction = (
            self.Kp * error +
            self.Ki * self._integral +
            self.Kd * derivative
        )

        # 应用修正得到实际速度
        velocity = self._prev_velocity + correction
        
        # 限幅到有效范围
        velocity = np.clip(velocity, -1.0, 1.0)
        
        self._prev_velocity = velocity

        return velocity.astype(np.float32)
