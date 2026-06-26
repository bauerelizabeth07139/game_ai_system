"""语义标签生成器 - 将数值转为自然语言标签"""
import logging
import math
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SemanticLabeler:
    """语义标签生成器"""

    # 威胁距离阈值（归一化坐标）
    CLOSE_RANGE = 0.3

    def __init__(self, config: dict = None):
        if config:
            self.CLOSE_RANGE = config.get('close_range', 0.3)

    def hp_status(self, hp: float) -> str:
        """血量状态标签：>70%健康，30%-70%受伤，<30%残血"""
        if hp > 0.7:
            return "健康"
        elif hp >= 0.3:
            return "受伤"
        else:
            return "残血(斩杀线)"

    def calculate_direction(self, dx: float, dy: float) -> str:
        """基于相对坐标计算方位

        Args:
            dx: 水平方向偏移（正=右，负=左）
            dy: 垂直方向偏移（正=下，负=上）

        Returns:
            方位字符串如 "正前方"、"左前方" 等
        """
        angle = math.degrees(math.atan2(-dy, dx))  # 上为0度

        if -22.5 <= angle < 22.5:
            return "正右方"
        elif 22.5 <= angle < 67.5:
            return "右前方"
        elif 67.5 <= angle < 112.5:
            return "正前方"
        elif 112.5 <= angle < 157.5:
            return "左前方"
        elif 157.5 <= angle or angle < -157.5:
            return "正左方"
        elif -157.5 <= angle < -112.5:
            return "左后方"
        elif -112.5 <= angle < -67.5:
            return "正后方"
        elif -67.5 <= angle < -22.5:
            return "右后方"
        return "正前方"

    def threat_level(self, hp: float, distance: float) -> str:
        """威胁等级评估"""
        if hp < 0.3 and distance < self.CLOSE_RANGE:
            return "高威胁-集火目标"
        elif hp < 0.3:
            return "低血量-优先目标"
        elif distance < self.CLOSE_RANGE:
            return "近距离-注意规避"
        else:
            return "一般"

    def generate_frame_data(self, frame_id: int, enemies: list, allies: list,
                           image_size: tuple = (480, 270)) -> dict:
        """生成当前帧结构化数据

        Returns:
            dict: {
                'frame_id': int,
                'enemies': [{'id': ..., 'hp': ..., 'hp_status': ..., 'pos': ..., 'tags': ...}],
                'allies': [{'id': ..., 'hp': ..., 'hp_status': ..., 'pos': ...}],
                'summary': str
            }
        """
        w, h = image_size if image_size else (480, 270)
        center_x, center_y = w / 2, h / 2

        enemy_list = []
        for i, e in enumerate(enemies):
            cx, cy = e.get('center', (0, 0))
            # 归一化相对坐标
            rel_x = (cx - center_x) / w
            rel_y = (cy - center_y) / h
            distance = math.sqrt(rel_x**2 + rel_y**2)

            hp = e.get('health', 0.0)
            enemy_list.append({
                'id': f'E{i+1}',
                'hp': round(hp, 2),
                'hp_status': self.hp_status(hp),
                'pos': self.calculate_direction(rel_x, rel_y),
                'tags': self.threat_level(hp, distance)
            })

        ally_list = []
        for i, a in enumerate(allies):
            cx, cy = a.get('center', (0, 0))
            rel_x = (cx - center_x) / w
            rel_y = (cy - center_y) / h

            hp = a.get('health', 0.0)
            ally_list.append({
                'id': f'A{i+1}',
                'hp': round(hp, 2),
                'hp_status': self.hp_status(hp),
                'pos': self.calculate_direction(rel_x, rel_y)
            })

        # 生成摘要
        summary = self._generate_summary(frame_id, enemy_list, ally_list)

        return {
            'frame_id': frame_id,
            'enemies': enemy_list,
            'allies': ally_list,
            'summary': summary
        }

    def _generate_summary(self, frame_id: int, enemies: list, allies: list) -> str:
        """生成当前帧自然语言摘要"""
        enemy_count = len(enemies)
        ally_count = len(allies)

        parts = [f"帧{frame_id}:"]

        if enemy_count == 0:
            parts.append("未检测到敌人")
        else:
            low_hp_enemies = [e for e in enemies if e['hp'] < 0.3]
            high_threat = [e for e in enemies if '高威胁' in e.get('tags', '')]
            parts.append(f"检测到{enemy_count}个敌人")
            if low_hp_enemies:
                parts.append(f"{len(low_hp_enemies)}个残血")
            if high_threat:
                parts.append(f"{len(high_threat)}个高威胁目标")

        if ally_count > 0:
            low_hp_allies = [a for a in allies if a['hp'] < 0.2]
            parts.append(f"友方{ally_count}人")
            if low_hp_allies:
                parts.append(f"{len(low_hp_allies)}人危急")

        return "，".join(parts)

    def generate_history_text(self, frames_data: list, offsets: list) -> str:
        """根据传入偏移列表提取各帧摘要生成描述过去变化的历史文本

        Args:
            frames_data: 各帧的语义数据列表
            offsets: 对应的偏移列表

        Returns:
            str: 历史趋势描述文本
        """
        if not frames_data:
            return "暂无历史数据"

        summaries = []
        for i, (data, offset) in enumerate(zip(frames_data, offsets)):
            if data and 'summary' in data:
                summaries.append(f"[{offset}帧]{data['summary']}")
            elif data and 'frame_id' in data:
                summaries.append(f"[{offset}帧]帧{data['frame_id']}数据")
            else:
                summaries.append(f"[{offset}帧]无数据")

        if not summaries:
            return "暂无历史数据"

        # 简单趋势分析
        trend = self._analyze_trend(frames_data)

        return "；".join(summaries) + f"。趋势：{trend}"

    def _analyze_trend(self, frames_data: list) -> str:
        """分析历史趋势"""
        valid_frames = [f for f in frames_data if f and 'enemies' in f]
        if len(valid_frames) < 2:
            return "数据不足"

        first = valid_frames[0]
        last = valid_frames[-1]

        enemy_change = len(last.get('enemies', [])) - len(first.get('enemies', []))

        if enemy_change > 0:
            return "敌方增援"
        elif enemy_change < 0:
            return "敌方减员"
        else:
            return "态势稳定"
