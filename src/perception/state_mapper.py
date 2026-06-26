"""状态映射器 - 将检测结果整理为标准化状态向量

功能：
1. 空间距离匹配：将检测到的目标和状态一一对应
2. LLM修正机制：按键触发，1分钟5次，累积上下文
3. 通知系统：从OCR识别大字号游戏通知
"""

import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

# 游戏类型状态字段映射（默认规则）
DEFAULT_FIELD_MAPPINGS = {
    'fps': {
        'hp': 'health',       # 生命值
        'mp': 'armor',        # 护甲
        'ammo': 'ammo',       # 弹药
        'position': 'pos',    # 位置
    },
    'fighting': {
        'hp': 'vitality',     # 生命值
        'mp': 'super_meter',  # 超必杀槽
        'stun': 'stun',       # 晕值
        'position': 'pos',
    },
    'moba': {
        'hp': 'health',
        'mp': 'mana',         # 法力值
        'level': 'level',
        'gold': 'gold',
        'position': 'pos',
    },
    'open_world': {
        'hp': 'health',
        'mp': 'endurance',    # 体力
        'hunger': 'hunger',
        'position': 'pos',
    },
    'unknown': {
        'hp': 'hp',
        'mp': 'mp',
        'position': 'pos',
    },
}


@dataclass
class GameObject:
    """游戏对象（目标）"""
    obj_id: str = ""           # 对象ID (E1, E2, A1, A2, Self)
    obj_type: str = "unknown"  # self/enemy/ally/object
    hp: float = 0.0            # 生命值 [0,1]
    mp: float = 0.0            # 蓝量/能量 [0,1]
    position: Tuple[float, float] = (0.0, 0.0)  # 归一化坐标 (x, y)
    distance: float = 0.0      # 到自身的距离
    is_alive: bool = True
    is_visible: bool = True
    # 匹配用特征
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 边界框 (x1,y1,x2,y2)
    color_class: str = ""      # red/green/unknown
    area: float = 0.0          # 轮廓面积
    extra: Dict = field(default_factory=dict)


@dataclass
class GameNotification:
    """游戏系统通知（通过大字号识别）"""
    text: str = ""
    font_size: float = 0.0     # 估算字号（像素高度）
    confidence: float = 0.0    # OCR置信度
    position: Tuple[int, int] = (0, 0)  # 位置
    timestamp: float = 0.0     # 时间戳
    frame_id: int = 0


@dataclass
class LLMMappingState:
    """LLM修正状态"""
    active: bool = False           # 是否激活
    start_time: float = 0.0        # 开始时间
    call_count: int = 0            # 已调用次数
    max_calls: int = 5             # 最大调用次数
    duration: float = 60.0         # 持续时间（秒）
    history: List[Dict] = field(default_factory=list)  # 历史输入输出


class StateMapper:
    """状态映射器"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.game_type = config.get('game_type', 'unknown')
        self.field_mappings = DEFAULT_FIELD_MAPPINGS.get(
            self.game_type, DEFAULT_FIELD_MAPPINGS['unknown']
        ).copy()
        self.max_enemies = 5
        self.max_allies = 5
        self.screen_width = config.get('screen_width', 960)
        self.screen_height = config.get('screen_height', 540)

        # 通知系统
        self.notifications: deque = deque(maxlen=20)
        self.min_font_size = config.get('min_notification_font_size', 20)  # 最小字号阈值

        # LLM修正状态
        self.llm_state = LLMMappingState()

        # 上一帧的目标（用于跟踪匹配）
        self._prev_enemies: List[GameObject] = []
        self._prev_allies: List[GameObject] = []

        logger.info(f"StateMapper initialized: game_type={self.game_type}")

    def set_game_type(self, game_type: str):
        """设置游戏类型"""
        self.game_type = game_type
        self.field_mappings = DEFAULT_FIELD_MAPPINGS.get(
            game_type, DEFAULT_FIELD_MAPPINGS['unknown']
        ).copy()

    # ═══════════════════════════════════════════════════════
    # LLM修正机制
    # ═══════════════════════════════════════════════════════

    def activate_llm_correction(self):
        """按键触发LLM修正（1分钟5次）"""
        self.llm_state.active = True
        self.llm_state.start_time = time.time()
        self.llm_state.call_count = 0
        self.llm_state.history = []
        logger.info("LLM correction activated for 60s (5 calls)")

    def is_llm_correction_active(self) -> bool:
        """检查LLM修正是否激活"""
        if not self.llm_state.active:
            return False
        elapsed = time.time() - self.llm_state.start_time
        if elapsed > self.llm_state.duration:
            self.deactivate_llm_correction()
            return False
        if self.llm_state.call_count >= self.llm_state.max_calls:
            self.deactivate_llm_correction()
            return False
        return True

    def deactivate_llm_correction(self):
        """停用LLM修正，清除缓冲区"""
        self.llm_state.active = False
        self.llm_state.history = []
        logger.info("LLM correction deactivated, buffer cleared")

    def get_llm_correction_context(self) -> List[Dict]:
        """获取LLM修正的累积上下文"""
        return self.llm_state.history.copy()

    def record_llm_correction(self, input_data: Dict, output_data: Dict):
        """记录LLM修正的输入输出"""
        self.llm_state.call_count += 1
        self.llm_state.history.append({
            'call': self.llm_state.call_count,
            'input': input_data,
            'output': output_data,
            'timestamp': time.time()
        })
        logger.info(f"LLM correction call {self.llm_state.call_count}/{self.llm_state.max_calls}")

        # 第5次结束后自动清除
        if self.llm_state.call_count >= self.llm_state.max_calls:
            self.deactivate_llm_correction()

    def update_field_mapping(self, original_name: str, mapped_name: str):
        """更新字段映射"""
        if original_name in self.field_mappings:
            old_name = self.field_mappings[original_name]
            self.field_mappings[original_name] = mapped_name
            logger.info(f"Field mapping: {original_name} -> {mapped_name} (was {old_name})")

    # ═══════════════════════════════════════════════════════
    # 通知系统（识别游戏大字号通知）
    # ═══════════════════════════════════════════════════════

    def extract_notifications_from_ocr(self, ocr_results: List[Dict], frame_id: int = 0) -> List[GameNotification]:
        """从OCR结果中识别大字号游戏通知

        游戏通知特征：
        1. 字号大（文本框高度 > 阈值）
        2. 通常在屏幕上方或中央
        3. 短文本（< 20字符）

        Args:
            ocr_results: OCR结果列表 [{'text': str, 'bbox': [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 'confidence': float}]
            frame_id: 帧序号

        Returns:
            List[GameNotification]: 识别出的通知列表
        """
        notifications = []

        for result in ocr_results:
            text = result.get('text', '').strip()
            if not text:
                continue

            bbox = result.get('bbox', [])
            if len(bbox) < 4:
                continue

            # 计算文本框高度（估算字号）
            y_values = [p[1] for p in bbox]
            font_height = max(y_values) - min(y_values)

            # 计算文本框中心位置
            x_center = sum(p[0] for p in bbox) / 4
            y_center = sum(p[1] for p in bbox) / 4

            # 判断是否为通知（大字号 + 短文本）
            if font_height >= self.min_font_size and len(text) < 30:
                notif = GameNotification(
                    text=text,
                    font_size=font_height,
                    confidence=result.get('confidence', 0.0),
                    position=(int(x_center), int(y_center)),
                    timestamp=time.time(),
                    frame_id=frame_id
                )
                notifications.append(notif)
                self.notifications.append(notif)
                logger.info(f"Notification detected: '{text}' (size={font_height:.0f}px)")

        return notifications

    def get_recent_notifications(self, count: int = 5) -> List[GameNotification]:
        """获取最近的通知"""
        return list(self.notifications)[-count:]

    def get_notifications_text(self) -> str:
        """获取通知文本（用于决策层）"""
        recent = self.get_recent_notifications(5)
        if not recent:
            return "No notifications"
        lines = []
        for n in recent:
            lines.append(f"[F{n.frame_id}] {n.text}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # 空间距离匹配（状态和物体一一对应）
    # ═══════════════════════════════════════════════════════

    def match_objects(self, current_enemies: List[Dict], current_allies: List[Dict],
                      self_position: Tuple[float, float] = (0.5, 0.5)) -> Tuple[List[GameObject], List[GameObject]]:
        """空间距离匹配：将当前检测结果和上一帧的目标一一对应

        匹配策略：
        1. 计算当前检测和上一帧目标的距离矩阵
        2. 使用贪心算法找最近匹配
        3. 未匹配的新目标分配新ID
        4. 未匹配的旧目标标记为丢失

        Args:
            current_enemies: 当前帧检测到的敌人
            current_allies: 当前帧检测到的友军
            self_position: 自身位置

        Returns:
            Tuple[List[GameObject], List[GameObject]]: 匹配后的敌人和友军列表
        """
        # 处理敌人
        matched_enemies = self._match_targets(
            current_enemies, self._prev_enemies, "enemy", self_position
        )

        # 处理友军
        matched_allies = self._match_targets(
            current_allies, self._prev_allies, "ally", self_position
        )

        # 更新上一帧
        self._prev_enemies = matched_enemies
        self._prev_allies = matched_allies

        return matched_enemies, matched_allies

    def _match_targets(self, current: List[Dict], previous: List[GameObject],
                       target_type: str, self_pos: Tuple[float, float]) -> List[GameObject]:
        """匹配目标列表"""
        if not current:
            return []

        # 构建当前目标
        current_objects = []
        for i, raw in enumerate(current):
            cx, cy = raw.get('center', (0, 0))
            norm_x = cx / self.screen_width
            norm_y = cy / self.screen_height

            obj = GameObject(
                obj_id="",  # 待分配
                obj_type=target_type,
                hp=raw.get('health', 0.0),
                mp=raw.get('mp', 0.0),
                position=(norm_x, norm_y),
                is_alive=True,
                is_visible=True,
                bbox=raw.get('bbox', (0, 0, 0, 0)),
                color_class=raw.get('color_class', ''),
                area=raw.get('area', 0.0)
            )

            # 计算距离
            dx = norm_x - self_pos[0]
            dy = norm_y - self_pos[1]
            obj.distance = np.sqrt(dx * dx + dy * dy)

            current_objects.append(obj)

        if not previous:
            # 没有上一帧，直接分配新ID
            prefix = "E" if target_type == "enemy" else "A"
            for i, obj in enumerate(current_objects):
                obj.obj_id = f"{prefix}{i+1}"
            return current_objects[:self.max_enemies if target_type == "enemy" else self.max_allies]

        # 计算距离矩阵
        cost_matrix = np.zeros((len(current_objects), len(previous)))
        for i, curr in enumerate(current_objects):
            for j, prev in enumerate(previous):
                dx = curr.position[0] - prev.position[0]
                dy = curr.position[1] - prev.position[1]
                cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

        # 贪心匹配
        matched_ids = {}
        used_prev = set()
        threshold = 0.2  # 距离阈值

        # 按距离排序匹配
        while True:
            if cost_matrix.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            i, j = min_idx
            if cost_matrix[i, j] > threshold:
                break

            # 匹配成功
            current_objects[i].obj_id = previous[j].obj_id
            matched_ids[i] = j
            used_prev.add(j)

            # 标记为已使用
            cost_matrix[i, :] = float('inf')
            cost_matrix[:, j] = float('inf')

        # 未匹配的当前目标分配新ID
        prefix = "E" if target_type == "enemy" else "A"
        existing_ids = {obj.obj_id for obj in previous}
        new_id_counter = 1
        for i, obj in enumerate(current_objects):
            if i not in matched_ids:
                # 生成新ID
                while f"{prefix}{new_id_counter}" in existing_ids:
                    new_id_counter += 1
                obj.obj_id = f"{prefix}{new_id_counter}"
                existing_ids.add(obj.obj_id)
                new_id_counter += 1

        # 限制数量
        max_count = self.max_enemies if target_type == "enemy" else self.max_allies
        return current_objects[:max_count]

    # ═══════════════════════════════════════════════════════
    # 主处理流程
    # ═══════════════════════════════════════════════════════

    def process(self, detection: Dict, ocr_results: List[Dict] = None,
                frame_id: int = 0) -> Dict:
        """处理检测结果

        Args:
            detection: ColorEngine的检测结果
            ocr_results: OCR结果（用于识别通知）
            frame_id: 帧序号

        Returns:
            Dict: {
                'self': GameObject,
                'enemies': List[GameObject],
                'allies': List[GameObject],
                'notifications': List[GameNotification],
                'field_mappings': Dict
            }
        """
        # 提取自身状态
        self_state = GameObject(
            obj_id="Self",
            obj_type="self",
            hp=0.8,
            mp=0.6,
            position=(0.5, 0.5),
            distance=0.0,
            is_alive=True,
            is_visible=True
        )

        # 空间距离匹配
        enemies_raw = detection.get('enemies', [])
        allies_raw = detection.get('allies', [])
        matched_enemies, matched_allies = self.match_objects(enemies_raw, allies_raw)

        # 识别游戏通知
        notifications = []
        if ocr_results:
            notifications = self.extract_notifications_from_ocr(ocr_results, frame_id)

        return {
            'self': self_state,
            'enemies': matched_enemies,
            'allies': matched_allies,
            'notifications': notifications,
            'field_mappings': self.field_mappings.copy()
        }

    def get_state_summary(self, result: Dict) -> str:
        """生成状态摘要"""
        self_state = result['self']
        enemies = result['enemies']
        allies = result['allies']

        lines = []
        lines.append(f"Self: {self_state.obj_id} HP={self_state.hp:.0%}")

        if enemies:
            enemy_strs = [f"{e.obj_id}:{e.hp:.0%}" for e in enemies[:3]]
            lines.append(f"Enemies({len(enemies)}): {', '.join(enemy_strs)}")
        else:
            lines.append("Enemies(0)")

        if allies:
            ally_strs = [f"{a.obj_id}:{a.hp:.0%}" for a in allies[:3]]
            lines.append(f"Allies({len(allies)}): {', '.join(ally_strs)}")
        else:
            lines.append("Allies(0)")

        return "\n".join(lines)
