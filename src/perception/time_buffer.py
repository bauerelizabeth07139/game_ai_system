"""时间切片缓冲器 - 环形缓冲区存储感知结果"""
import logging
import threading
from collections import deque
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TimeBuffer:
    """环形缓冲区，存储最近N帧的感知结果

    提供两个独立采样接口：
    - 策略层采样：帧偏移 [0, -2, -6] -> 141*3=423维
    - 决策层采样：帧偏移 [0, -8, -42]
    """

    # 默认帧偏移
    MLP_OFFSETS = [0, -2, -6]  # 3个时间点 -> 423维
    DECISION_OFFSETS = [0, -8, -42]

    def __init__(self, config: dict = None):
        self.buffer_size = 50
        self.mlp_offsets = self.MLP_OFFSETS
        self.decision_offsets = self.DECISION_OFFSETS

        if config:
            self.buffer_size = config.get('buffer_size', 50)
            self.mlp_offsets = config.get('mlp_history_frames', self.MLP_OFFSETS)
            self.decision_offsets = config.get('decision_history_frames', self.DECISION_OFFSETS)

        self._buffer: deque = deque(maxlen=self.buffer_size)
        self._lock = threading.Lock()
        self._frame_counter = 0

    def push(self, perception_data: dict):
        """推入一帧感知结果

        Args:
            perception_data: 包含 'frame_id', 'state_vector'(128维), 'semantic_data' 等
        """
        with self._lock:
            if 'frame_id' not in perception_data:
                perception_data['frame_id'] = self._frame_counter
            self._buffer.append(perception_data)
            self._frame_counter += 1

    def get_latest(self) -> Optional[dict]:
        """获取最新一帧"""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def get_by_offset(self, offset: int) -> Optional[dict]:
        """根据帧偏移获取数据，负数表示往前偏移

        offset=0 表示最新帧，-1表示上一帧，以此类推
        若偏移超出范围则返回最近可用帧
        """
        with self._lock:
            if not self._buffer:
                return None

            # 最新帧索引为 len-1，offset=0 => 最新帧，-1 => 上一帧
            idx = len(self._buffer) - 1 + offset
            if idx < 0:
                # 超出范围，返回最早可用帧
                return self._buffer[0]
            if idx >= len(self._buffer):
                # 超出范围，返回最新帧
                return self._buffer[-1]
            return self._buffer[idx]

    def sample_for_mlp(self) -> np.ndarray:
        """策略层采样：返回帧偏移 [0,-2,-6] 的状态向量拼接

        Returns:
            np.ndarray: shape (423,) 即 141*3 维的状态向量
        """
        state_dim = 141
        states = []

        for offset in self.mlp_offsets:
            data = self.get_by_offset(offset)
            if data is not None and 'state_vector' in data:
                sv = np.array(data['state_vector'], dtype=np.float32)
                if sv.shape[0] < state_dim:
                    # 填充零向量
                    sv = np.pad(sv, (0, state_dim - sv.shape[0]))
                elif sv.shape[0] > state_dim:
                    sv = sv[:state_dim]
                states.append(sv)
            else:
                # 填充零向量
                states.append(np.zeros(state_dim, dtype=np.float32))

        return np.concatenate(states)

    def sample_for_decision(self) -> List[dict]:
        """决策层采样：返回帧偏移 [0,-4,-12,-24,-40] 的原始摘要数据

        Returns:
            List[dict]: 各帧的语义数据列表
        """
        results = []
        for offset in self.decision_offsets:
            data = self.get_by_offset(offset)
            if data is not None:
                results.append(data.get('semantic_data', {}))
            else:
                results.append({})
        return results

    def __len__(self):
        with self._lock:
            return len(self._buffer)

    @property
    def current_frame_id(self) -> int:
        return self._frame_counter
