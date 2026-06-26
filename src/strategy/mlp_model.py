"""策略层MLP模型 - 4层全连接网络"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# 策略指令到索引的映射
STRATEGY_TO_IDX = {
    'FOCUS_FIRE': 0, 'RETREAT': 1, 'FLANK': 2, 'DEFEND': 3, 'PUSH': 4,
    'HOLD': 5, 'ROTATE': 6, 'ENGAGE': 7, 'DISENGAGE': 8
}
IDX_TO_STRATEGY = {v: k for k, v in STRATEGY_TO_IDX.items()}

# 游戏类型到索引的映射
GAME_TYPE_TO_IDX = {
    'unknown': 0, 'fps': 1, 'fighting': 2, 'moba': 3, 'open_world': 4, 'auto': 5
}
IDX_TO_GAME_TYPE = {v: k for k, v in GAME_TYPE_TO_IDX.items()}

# 游戏类型专属状态范围 [start, end)
GAME_TYPE_STATE_RANGES = {
    'fps':        (50, 75),   # [50-74] 25维
    'fighting':   (75, 97),   # [75-96] 22维
    'moba':       (97, 119),  # [97-118] 22维
    'open_world': (119, 141), # [119-140] 22维
}

# 40维输出的动作名称（按键盘排布）
ACTION_NAMES = [
    # 0-3: 连续控制（Sigmoid输出后映射到[-1,1]）
    'mouse_vx', 'mouse_vy', 'move_vx', 'move_vy',
    # 4-6: 鼠标按键
    'LMB', 'RMB', 'MMB',
    # 7-10: 移动键
    'W', 'A', 'S', 'D',
    # 11-15: 功能键
    'Space', 'Shift', 'Ctrl', 'Tab', 'Esc',
    # 16-25: 数字键
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    # 26-35: 技能/动作键
    'Q', 'E', 'R', 'F', 'G', 'H', 'Z', 'X', 'C', 'V',
    # 36-39: 其他键
    'B', 'N', 'T', 'M'
]

# GUI键盘布局（用于显示）
KEYBOARD_LAYOUT = [
    ['Esc', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['Tab', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['Ctrl', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Shift', 'Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['Space'],
    ['LMB', 'RMB', 'MMB'],
]


class MLPModel(nn.Module):
    """4层全连接策略网络

    输入：状态向量(141*3=423维) + 指令嵌入(32维) + 游戏类型(1维) = 456维
    输出：控制原语(40维) 全部Sigmoid [0,1]
        0-3: 连续控制（鼠标/移动速度）
        4-39: 36个按键概率
    """

    INPUT_DIM = 456    # 141*3 + 32 + 1
    STATE_DIM = 141
    NUM_TIME_POINTS = 3
    EMBED_DIM = 32
    NUM_STRATEGIES = 9
    NUM_GAME_TYPES = 6  # unknown, fps, fighting, moba, open_world, auto
    OUTPUT_DIM = 40

    def __init__(self):
        super().__init__()

        # 指令嵌入层
        self.strategy_embedding = nn.Embedding(self.NUM_STRATEGIES, self.EMBED_DIM)

        # 游戏类型嵌入（1维标量输入，直接拼接）
        # 不需要嵌入，直接用标量

        # 4层全连接网络：512 -> 1024 -> 512 -> 40
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.OUTPUT_DIM),
            nn.Sigmoid()  # 全部Sigmoid输出
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, state_vectors: torch.Tensor, strategy_idx: torch.Tensor,
                game_type: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            state_vectors: shape (batch, 384) - 3个时间点的状态向量拼接
            strategy_idx: shape (batch,) - 策略索引
            game_type: shape (batch, 1) - 游戏类型标量 [0-4]

        Returns:
            torch.Tensor: shape (batch, 40) - 控制原语，全部Sigmoid [0,1]
        """
        # 获取指令嵌入
        embed = self.strategy_embedding(strategy_idx)  # (batch, 32)

        # 拼接输入
        x = torch.cat([state_vectors, embed, game_type], dim=-1)  # (batch, 417)

        # 前向传播
        output = self.network(x)  # (batch, 40) 全部Sigmoid

        return output

    def get_default_state_dict(self) -> dict:
        """获取默认模型状态（用于保存/加载）"""
        return self.state_dict()
