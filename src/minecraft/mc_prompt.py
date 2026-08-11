import json
from typing import Dict, List

from src.minecraft.mc_perception import MCWorldState


class MCPromptBuilder:
    def __init__(self):
        self.strategy_prompts = {
            'GATHER_WOOD': (
                "收集木材：寻找樹木并破壞原木方塊。"
                "優先收集至少4個原木，將其製作成木板和木棍。"
                "使用斧頭可以提高效率。"
            ),
            'CRAFT_TOOLS': (
                "製作工具：將原木製作成木板，用木板製作工作台。"
                "使用工作台製作木鎬、木斧和木劍。"
                "優先製作木鎬以準備挖石。"
            ),
            'MINE_STONE': (
                "挖掘石頭：使用木鎬挖掘石頭獲取圓石。"
                "收集足夠的圓石製作石製工具和熔爐。"
                "小心附近的洞穴和怪物。"
            ),
            'FIND_FOOD': (
                "尋找食物：收集食物以維持飢餓值。"
                "可以殺動物獲取肉類，或收集蘋果、麵包。"
                "用熔爐烹飪生肉可以獲得更多飢餓回復。"
            ),
            'MINE_IRON': (
                "挖掘鐵礦：在地下尋找鐵礦石。"
                "使用石鎬或更好的鎬子挖掘鐵礦。"
                "用熔爐將鐵礦石熔煉成鐵錠。需要收集至少3個鐵錠。"
            ),
            'CRAFT_ARMOR': (
                "製作盔甲：使用鐵錠製作鐵製盔甲。"
                "優先製作胸甲和護腿。"
                "穿上盔甲可以提高生存能力。"
            ),
            'MINE_DIAMOND': (
                "挖掘鑽石：在Y=-54左右的深層尋找鑽石礦。"
                "使用鐵鎬或更好的鎬子挖掘鑽石。"
                "收集至少3顆鑽石以準備製作鑽石鎬。"
            ),
            'FIND_LAVA': (
                "尋找岩漿：在深層或下界尋找岩漿池。"
                "需要製作水桶來將岩漿轉換為黑曜石。"
                "收集至少10個黑曜石來建造地獄傳送門。"
            ),
            'BUILD_PORTAL': (
                "建造傳送門：使用黑曜石建造4x5的地獄傳送門框架。"
                "用打火石點燃傳送門。"
                "進入傳送門前往下界。"
            ),
            'ENTER_NETHER': (
                "進入地獄：通過傳送門進入下界。"
                "在下界中尋找下界堡壘。"
                "小心下界的危險生物。"
            ),
            'FIND_FORTRESS': (
                "尋找堡壘：在下界中尋找下界堡壘。"
                "堡壘通常在下界荒地中生生成。"
                "尋找烈焰神生成籠。"
            ),
            'KILL_BLAZE': (
                "擊殺烈焰神：在下界堡壘中尋找並擊殺烈焰神。"
                "使用弓或劍攻擊烈焰神。"
                "收集至少3根烈焰棒。"
            ),
            'FIND_END_PORTAL': (
                "尋找終界傳送門：使用烈焰粉和終界珍珠合成終界之眼。"
                "使用終界之眼尋找要塞中的終界傳送門。"
                "激活傳送門進入終界。"
            ),
            'KILL_ENDER_DRAGON': (
                "擊殺終界龍：在終界中先摧毀終界水晶。"
                "使用弓攻擊終界水晶和終界龍。"
                "終界龍停在中間柱子時用劍攻擊。"
                "擊殺終界龍即可通關！"
            ),
        }

    def build_system_prompt(self, goal: str) -> str:
        base = (
            "你是一個Minecraft AI玩家。你的目標是通關Minecraft。\n"
            "你目前正在執行的子目標是：" + goal + "\n\n"
        )

        goal_desc = self.strategy_prompts.get(goal, "探索世界並收集資源。")
        base += goal_desc + "\n\n"

        base += (
            "請根據當前狀態給出決策。\n"
            "輸出格式：STRATEGY|TARGET|PRIORITY|REASON\n"
            "可選策略：EXPLORE, MINE, ATTACK, FLEE, CRAFT, BUILD, FARM, TRADE, SLEEP, EAT\n"
        )
        return base

    def build_user_prompt(self, state: MCWorldState, history: List[Dict]) -> str:
        parts = []

        parts.append(f"HP:{state.health:.0f}/{state.max_health:.0f}")
        parts.append(f"HUNGER:{state.hunger:.0f}")
        parts.append(f"POS:({state.pos_x:.0f},{state.pos_y:.0f},{state.pos_z:.0f})")
        parts.append(f"DIM:{state.dimension.split(':')[-1]}")
        parts.append(f"GOAL:{state.current_goal}")
        parts.append(f"COMBAT:{'YES' if state.in_combat else 'NO'}")
        parts.append(f"HOSTILES:{state.nearby_hostiles}")
        parts.append(f"TIME:{'NIGHT' if state.is_night else 'DAY'}")
        parts.append(f"ITEMS:{len(state.inventory)}")

        if state.block_looking_at:
            parts.append(f"LOOK:{state.block_looking_at.split(':')[-1]}")

        inv_summary = []
        all_items = {}
        for entry in state.hotbar + state.inventory:
            name = entry.get('item', '').split(':')[-1]
            cnt = entry.get('count', 0)
            all_items[name] = all_items.get(name, 0) + cnt

        for name, cnt in sorted(all_items.items(), key=lambda x: -x[1])[:10]:
            inv_summary.append(f"{name}x{cnt}")
        parts.append(f"INV:{','.join(inv_summary)}")

        return " | ".join(parts) + "\n"

    def build_state_text(self, state: MCWorldState) -> str:
        return self.build_user_prompt(state, [])

    def parse_decision(self, response: str) -> Dict:
        response = response.strip()
        try:
            data = json.loads(response)
            return {
                'strategy': data.get('strategy', 'EXPLORE'),
                'target': data.get('target', ''),
                'priority': data.get('priority', 'medium'),
                'reasoning': data.get('reasoning', ''),
            }
        except json.JSONDecodeError:
            pass

        if '|' in response:
            parts = [p.strip() for p in response.split('|')]
            return {
                'strategy': parts[0] if len(parts) > 0 else 'EXPLORE',
                'target': parts[1] if len(parts) > 1 else '',
                'priority': parts[2] if len(parts) > 2 else 'medium',
                'reasoning': parts[3] if len(parts) > 3 else '',
            }

        valid_strategies = ['EXPLORE', 'MINE', 'ATTACK', 'FLEE', 'CRAFT',
                           'BUILD', 'FARM', 'TRADE', 'SLEEP', 'EAT']
        for strategy in valid_strategies:
            if strategy in response.upper():
                return {
                    'strategy': strategy,
                    'target': '',
                    'priority': 'medium',
                    'reasoning': response,
                }

        return {
            'strategy': 'EXPLORE',
            'target': '',
            'priority': 'medium',
            'reasoning': 'Default fallback',
        }
