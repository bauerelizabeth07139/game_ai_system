"""策略层推理引擎"""
import logging
import time
import numpy as np
import torch
from .mlp_model import (MLPModel, STRATEGY_TO_IDX, IDX_TO_STRATEGY,
                         GAME_TYPE_TO_IDX, ACTION_NAMES)

logger = logging.getLogger(__name__)


class StrategyInference:
    """策略层推理引擎"""

    def __init__(self, config: dict = None):
        self.device = torch.device('cpu')
        self.model = MLPModel()
        self.model.eval()

        self._current_strategy_idx = STRATEGY_TO_IDX.get('HOLD', 5)
        self._current_strategy = 'HOLD'
        self._current_game_type = 0.0  # 默认fps=0

        self._inference_times = []

    def set_strategy(self, strategy: str):
        """设置当前策略"""
        if strategy in STRATEGY_TO_IDX:
            self._current_strategy = strategy
            self._current_strategy_idx = STRATEGY_TO_IDX[strategy]

    def set_game_type(self, game_type: str):
        """设置游戏类型"""
        if game_type in GAME_TYPE_TO_IDX:
            self._current_game_type = float(GAME_TYPE_TO_IDX[game_type])
            logger.info(f"游戏类型: {game_type} ({self._current_game_type})")

    def infer(self, state_vector: np.ndarray) -> np.ndarray:
        """运行推理

        Args:
            state_vector: shape (423,) - 3个时间点的状态向量拼接 (141*3)

        Returns:
            np.ndarray: shape (40,) - 控制原语，全部Sigmoid [0,1]
        """
        start = time.perf_counter()

        try:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            strategy_tensor = torch.LongTensor([self._current_strategy_idx]).to(self.device)
            game_type_tensor = torch.FloatTensor([[self._current_game_type]]).to(self.device)

            with torch.no_grad():
                output = self.model(state_tensor, strategy_tensor, game_type_tensor)

            result = output.squeeze(0).cpu().numpy()

            elapsed = time.perf_counter() - start
            self._inference_times.append(elapsed)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)

            return result

        except Exception as e:
            logger.error(f"策略推理异常: {e}")
            return np.zeros(40, dtype=np.float32)

    def get_performance_stats(self) -> dict:
        if not self._inference_times:
            return {'avg_ms': 0, 'max_ms': 0, 'min_ms': 0}
        times_ms = [t * 1000 for t in self._inference_times]
        return {
            'avg_ms': sum(times_ms) / len(times_ms),
            'max_ms': max(times_ms),
            'min_ms': min(times_ms)
        }

    @property
    def current_strategy(self) -> str:
        return self._current_strategy

    @property
    def current_game_type(self) -> str:
        return {v: k for k, v in GAME_TYPE_TO_IDX.items()}.get(int(self._current_game_type), 'fps')
