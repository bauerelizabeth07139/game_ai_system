"""提示词构建器 - 生成精简的LLM用户消息"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """提示词构建器 - 极简格式减少token消耗"""

    def __init__(self, config: dict = None):
        pass

    def build_user_message(self, frame_id: int, enemies: list, allies: list,
                          summary: str, history_text: str) -> str:
        """构建精简的用户消息

        格式：
        F:{frame_id}
        E:{enemy_count}[{enemy_details}]
        A:{ally_count}[{ally_details}]
        H:{history_brief}
        """
        # 敌方信息（精简）
        if enemies:
            enemy_parts = []
            for e in enemies[:3]:  # 最多3个
                hp = int(e.get('hp', 0) * 100)
                eid = e.get('id', '?')
                status = 'low' if hp < 30 else 'mid' if hp < 70 else 'high'
                enemy_parts.append(f"{eid}:{hp}%{status}")
            enemies_str = ",".join(enemy_parts)
        else:
            enemies_str = "none"

        # 友方信息（精简）
        if allies:
            ally_parts = []
            for a in allies[:3]:
                hp = int(a.get('hp', 0) * 100)
                aid = a.get('id', '?')
                status = 'low' if hp < 20 else 'mid' if hp < 70 else 'high'
                ally_parts.append(f"{aid}:{hp}%{status}")
            allies_str = ",".join(ally_parts)
        else:
            allies_str = "none"

        # 精简历史（只保留趋势）
        history_brief = ""
        if history_text and "趋势" in history_text:
            # 提取趋势部分
            parts = history_text.split("趋势：")
            if len(parts) > 1:
                history_brief = parts[-1][:20]
        if not history_brief:
            history_brief = "stable"

        message = (
            f"F:{frame_id}\n"
            f"E:{len(enemies)}[{enemies_str}]\n"
            f"A:{len(allies)}[{allies_str}]\n"
            f"H:{history_brief}\n"
            f"S:{summary[:30]}"
        )

        return message

    def build_from_semantic_data(self, semantic_data: dict, history_text: str) -> str:
        """从语义数据构建消息"""
        if not semantic_data:
            return "F:0\nE:0[none]\nA:0[none]\nH:no_data\nS:no_data"

        return self.build_user_message(
            frame_id=semantic_data.get('frame_id', 0),
            enemies=semantic_data.get('enemies', []),
            allies=semantic_data.get('allies', []),
            summary=semantic_data.get('summary', ''),
            history_text=history_text
        )
