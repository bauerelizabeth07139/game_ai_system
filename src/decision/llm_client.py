"""决策层LLM客户端 - 按游戏类型专业化"""
import json
import logging
import time
import threading
from typing import Optional
import requests

logger = logging.getLogger(__name__)

PROVIDER_URLS = {
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com/v1',
    'ollama': 'http://localhost:11434/v1',
    'vllm': 'http://localhost:8000/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4',
    'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'local': 'http://localhost:11434/v1'
}

DEFAULT_DECISION = {"strategy": "HOLD", "target": None, "priority": "low", "reasoning": "default"}

VALID_STRATEGIES = [
    "FOCUS_FIRE", "RETREAT", "FLANK", "DEFEND", "PUSH",
    "HOLD", "ROTATE", "ENGAGE", "DISENGAGE"
]

PRIORITY_MAP = {'0': 'low', '1': 'medium', '2': 'high',
                'low': 'low', 'medium': 'medium', 'high': 'high'}

# 游戏类型专业化提示词
# 注意：LLM是纯文本模型，接收的是结构化文本数据（非图像）
GAME_TYPE_PROMPTS = {
    'fps': (
        "You are a FPS game tactical decision AI. You receive TEXT data only (no images).\n"
        "Input format: Frame ID, enemy list [id, HP%, status, position], ally list, game state.\n"
        "You must analyze the NUMERICAL data to make decisions.\n"
        "Decision logic:\n"
        "- Enemy HP<30% -> FOCUS_FIRE (finish kill)\n"
        "- Self HP<20% or outnumbered -> RETREAT\n"
        "- Enemy count < ally count-1 -> PUSH\n"
        "- All enemies healthy, no advantage -> DEFEND/HOLD\n"
        "- Enemy distracted by allies -> FLANK\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: FOCUS_FIRE|E1|2|hp15%"
    ),
    'fighting': (
        "You are a fighting game tactical AI. You receive TEXT data only (no images).\n"
        "Input format: Self HP, opponent HP, distance, frame advantage, combo count, energy.\n"
        "You must analyze the NUMERICAL data to make decisions.\n"
        "Decision logic:\n"
        "- Opponent HP<30% -> ENGAGE (confirm kill)\n"
        "- Self HP<20% or cornered -> RETREAT\n"
        "- Frame advantage >0 -> ENGAGE (pressure)\n"
        "- Frame advantage <0 -> DEFEND (block)\n"
        "- Opponent in corner -> PUSH (pressure)\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: ENGAGE|null|2|frame_adv"
    ),
    'moba': (
        "You are a MOBA game tactical AI. You receive TEXT data only (no images).\n"
        "Input format: Level, gold, CS, KDA, skill CDs, tower HP, dragon timer.\n"
        "You must analyze the NUMERICAL data to make decisions.\n"
        "Decision logic:\n"
        "- Carry low HP in teamfight -> FOCUS_FIRE\n"
        "- Self HP<20% or jungler near -> RETREAT\n"
        "- Enemy dead/rotated, wave pushing -> PUSH\n"
        "- Dragon/Baron spawning -> ROTATE (objective)\n"
        "- Behind in gold -> HOLD (farm safe)\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: FLANK|null|2|adc_low"
    ),
    'open_world': (
        "You are an open world game AI. You receive TEXT data only (no images).\n"
        "Input format: HP, stamina, hunger, element status, boss HP, quest progress.\n"
        "You must analyze the NUMERICAL data to make decisions.\n"
        "Decision logic:\n"
        "- Boss HP<20% -> PUSH (finish)\n"
        "- Self HP<30% or stamina low -> RETREAT (heal)\n"
        "- Boss enraged -> DEFEND (wait opening)\n"
        "- Resource nearby -> ENGAGE (collect)\n"
        "- Exploring -> HOLD (observe)\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: DISENGAGE|null|1|heal"
    ),
    'unknown': (
        "You are a game AI. You receive TEXT data only (no images).\n"
        "Input format: HP values, positions, counts for enemies and allies.\n"
        "You must analyze the NUMERICAL data to make decisions.\n"
        "Decision logic:\n"
        "- Enemy HP<30% -> FOCUS_FIRE\n"
        "- Self HP<20% -> RETREAT\n"
        "- Enemy count < ally count -> PUSH\n"
        "- No clear advantage -> HOLD\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: HOLD|null|0|observe"
    ),
    'auto': (
        "You are a game AI. You receive TEXT data only (no images).\n"
        "Analyze the structure of the data to determine game type:\n"
        "- Ammo, armor, crosshair -> FPS\n"
        "- Frame advantage, combo, energy -> Fighting\n"
        "- Level, gold, CS, tower HP -> MOBA\n"
        "- Stamina, hunger, element -> Open World\n"
        "Then apply type-specific logic.\n"
        "Output ONLY: STRATEGY|TARGET|PRIORITY|REASON\n"
        "STRATEGIES: FOCUS_FIRE,RETREAT,FLANK,DEFEND,PUSH,HOLD,ROTATE,ENGAGE,DISENGAGE\n"
        "Example: HOLD|null|0|observe"
    ),
}

# 游戏类型检测提示词（基于文本数据特征判断）
GAME_TYPE_DETECT_PROMPT = (
    "You are a game type classifier. You receive TEXT data only (no images).\n"
    "Analyze the DATA STRUCTURE to determine game type:\n"
    "- If data contains ammo/armor/crosshair -> fps\n"
    "- If data contains frame_advantage/combo/energy -> fighting\n"
    "- If data contains level/gold/cs/tower_hp -> moba\n"
    "- If data contains stamina/hunger/element -> open_world\n"
    "Output ONLY one word: fps, fighting, moba, or open_world"
)

# 字段映射修正提示词（LLM只修正称呼，不改变维度）
FIELD_MAPPING_CORRECTION_PROMPT = (
    "You are a game data field name advisor.\n"
    "Given a game type and generic field names, suggest the correct in-game term.\n"
    "Rules:\n"
    "- Only suggest the CORRECT NAME for the field, not new fields\n"
    "- Do NOT change the number of fields or add new ones\n"
    "- Use common game terminology\n"
    "Input format: game_type|field1=generic_name,field2=generic_name,...\n"
    "Output format: field1=correct_name,field2=correct_name,...\n"
    "Examples:\n"
    "fps|hp=hp,mp=armor -> hp=health,mp=armor\n"
    "fighting|hp=hp,mp=energy -> hp=vitality,mp=super_meter\n"
    "moba|hp=hp,mp=mp -> hp=health,mp=mana\n"
    "open_world|hp=hp,mp=stamina -> hp=health,mp=endurance"
)


class LLMClient:
    """决策层LLM客户端"""

    def __init__(self, config: dict = None):
        self.api_type = 'openai'
        self.api_base = PROVIDER_URLS['openai']
        self.api_key = ''
        self.model_name = 'gpt-4o-mini'
        self.timeout = 10
        self.max_retries = 2
        self.interval_frames = 50
        self.history_frames = [0, -4, -12, -24, -40]
        self.game_type = 'auto'

        if config:
            self.api_type = config.get('api_type', self.api_type)
            self.api_base = config.get('api_base', self.api_base)
            self.api_key = config.get('api_key', self.api_key)
            self.model_name = config.get('model_name', self.model_name)
            self.timeout = config.get('timeout', self.timeout)
            self.max_retries = config.get('max_retries', self.max_retries)
            self.interval_frames = config.get('interval_frames', self.interval_frames)
            self.history_frames = config.get('history_frames', self.history_frames)
            self.game_type = config.get('game_type', self.game_type)

            if self.api_base in PROVIDER_URLS:
                self.api_base = PROVIDER_URLS[self.api_base]

        self._last_decision = DEFAULT_DECISION.copy()
        self._last_call_frame = -999
        self._lock = threading.Lock()
        self._pending = False
        self._detected_game_type = None

        # 设置系统提示词
        self._update_system_prompt()

    def _update_system_prompt(self):
        """根据游戏类型更新系统提示词"""
        gt = self._detected_game_type or self.game_type
        self.system_prompt = GAME_TYPE_PROMPTS.get(gt, GAME_TYPE_PROMPTS['auto'])

    def set_game_type(self, game_type: str):
        """设置游戏类型"""
        self.game_type = game_type
        self._detected_game_type = None
        self._update_system_prompt()

    def should_call(self, current_frame: int) -> bool:
        return (current_frame - self._last_call_frame) >= self.interval_frames

    def get_decision(self) -> dict:
        with self._lock:
            return self._last_decision.copy()

    def call_async(self, user_message: str, current_frame: int):
        if self._pending:
            return
        self._pending = True
        threading.Thread(target=self._call_api_thread,
                         args=(user_message, current_frame), daemon=True).start()

    def _call_api_thread(self, user_message: str, current_frame: int):
        try:
            for attempt in range(self.max_retries):
                try:
                    decision = self._make_request(user_message)
                    if decision:
                        with self._lock:
                            self._last_decision = decision
                            self._last_call_frame = current_frame
                        logger.info(f"决策更新: {decision['strategy']} (帧{current_frame})")
                        return
                except Exception as e:
                    logger.warning(f"API调用失败(尝试{attempt+1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)

            # 失败时使用模拟决策
            logger.warning("API调用失败，使用模拟决策")
            with self._lock:
                self._last_decision = DEFAULT_DECISION.copy()
                self._last_call_frame = current_frame
        finally:
            self._pending = False

    def _make_request(self, user_message: str) -> Optional[dict]:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            'temperature': 0.2,
            'max_tokens': 50
        }

        if self.api_type == 'anthropic':
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01'
            }
            payload = {
                'model': self.model_name,
                'max_tokens': 50,
                'system': self.system_prompt,
                'messages': [{'role': 'user', 'content': user_message}]
            }
            url = f"{self.api_base}/messages"
        else:
            url = f"{self.api_base}/chat/completions"

        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if self.api_type == 'anthropic':
            content = data.get('content', [{}])[0].get('text', '')
        else:
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        return self._parse_decision(content.strip())

    def detect_game_type(self, user_message: str) -> Optional[str]:
        """让大模型检测游戏类型"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            payload = {
                'model': self.model_name,
                'messages': [
                    {'role': 'system', 'content': GAME_TYPE_DETECT_PROMPT},
                    {'role': 'user', 'content': user_message}
                ],
                'temperature': 0.1,
                'max_tokens': 10
            }

            if self.api_type == 'anthropic':
                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                    'anthropic-version': '2023-06-01'
                }
                payload = {
                    'model': self.model_name,
                    'max_tokens': 10,
                    'system': GAME_TYPE_DETECT_PROMPT,
                    'messages': [{'role': 'user', 'content': user_message}]
                }
                url = f"{self.api_base}/messages"
            else:
                url = f"{self.api_base}/chat/completions"

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if self.api_type == 'anthropic':
                content = data.get('content', [{}])[0].get('text', '').strip().lower()
            else:
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip().lower()

            # 提取游戏类型
            for gt in ['fps', 'fighting', 'moba', 'open_world']:
                if gt in content:
                    self._detected_game_type = gt
                    self._update_system_prompt()
                    logger.info(f"检测到游戏类型: {gt}")
                    return gt

            return None
        except Exception as e:
            logger.warning(f"游戏类型检测失败: {e}")
            return None

    def _parse_decision(self, content: str) -> Optional[dict]:
        try:
            if '|' in content:
                parts = content.strip().split('|')
                if len(parts) >= 3:
                    strategy = parts[0].strip().upper()
                    target = parts[1].strip()
                    priority = PRIORITY_MAP.get(parts[2].strip().lower(), 'medium')
                    reasoning = parts[3].strip() if len(parts) > 3 else ''

                    if strategy in VALID_STRATEGIES:
                        return {
                            "strategy": strategy,
                            "target": target if target.lower() not in ('null', 'none', '') else None,
                            "priority": priority,
                            "reasoning": reasoning
                        }

            decision = json.loads(content)
            if 'strategy' in decision:
                if decision['strategy'] not in VALID_STRATEGIES:
                    decision['strategy'] = 'HOLD'
                decision.setdefault('target', None)
                decision.setdefault('priority', 'medium')
                decision.setdefault('reasoning', '')
                return decision

            return None
        except Exception:
            for strategy in VALID_STRATEGIES:
                if strategy in content.upper():
                    return {"strategy": strategy, "target": None, "priority": "medium", "reasoning": content[:20]}
            return None

    def simulate_decision(self, perception_data: dict = None) -> dict:
        """模拟决策"""
        if perception_data is None:
            return DEFAULT_DECISION.copy()

        enemies = perception_data.get('enemies', [])
        allies = perception_data.get('allies', [])

        low_hp_enemies = [e for e in enemies if e.get('hp', 1.0) < 0.3]
        if low_hp_enemies:
            target = min(low_hp_enemies, key=lambda e: e.get('hp', 1.0))
            return {"strategy": "FOCUS_FIRE", "target": target.get('id'), "priority": "high", "reasoning": "low_hp"}

        low_hp_allies = [a for a in allies if a.get('hp', 1.0) < 0.2]
        if len(low_hp_allies) >= 2:
            return {"strategy": "RETREAT", "target": None, "priority": "high", "reasoning": "ally_low"}

        if len(enemies) < len(allies) - 1:
            return {"strategy": "PUSH", "target": None, "priority": "medium", "reasoning": "advantage"}

        return {"strategy": "DEFEND", "target": None, "priority": "low", "reasoning": "neutral"}

    def correct_field_mapping(self, game_type: str, current_mappings: dict) -> Optional[dict]:
        """使用LLM修正字段映射（只修改称呼，不改变维度）

        Args:
            game_type: 游戏类型
            current_mappings: 当前字段映射 {'hp': 'hp', 'mp': 'armor', ...}

        Returns:
            dict: 修正后的映射 {'hp': 'health', 'mp': 'armor', ...} 或 None
        """
        if not self.api_key or self._failed:
            return None

        # 构建请求
        fields_str = ",".join(f"{k}={v}" for k, v in current_mappings.items())
        user_msg = f"{game_type}|{fields_str}"

        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_type == 'anthropic':
                headers['x-api-key'] = self.api_key
                headers['anthropic-version'] = '2023-06-01'
                payload = {
                    'model': self.model_name,
                    'max_tokens': 100,
                    'messages': [
                        {'role': 'user', 'content': f"{FIELD_MAPPING_CORRECTION_PROMPT}\n\n{user_msg}"}
                    ]
                }
            else:
                headers['Authorization'] = f'Bearer {self.api_key}'
                payload = {
                    'model': self.model_name,
                    'messages': [
                        {'role': 'system', 'content': FIELD_MAPPING_CORRECTION_PROMPT},
                        {'role': 'user', 'content': user_msg}
                    ],
                    'max_tokens': 100,
                    'temperature': 0.1
                }

            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=5
            )

            if resp.status_code != 200:
                logger.warning(f"Field mapping correction failed: {resp.status_code}")
                return None

            data = resp.json()
            if self.api_type == 'anthropic':
                content = data.get('content', [{}])[0].get('text', '').strip()
            else:
                content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            # 解析结果：field1=correct_name,field2=correct_name,...
            result = {}
            for pair in content.split(','):
                if '=' in pair:
                    key, val = pair.strip().split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    if key in current_mappings:
                        result[key] = val

            if result:
                logger.info(f"Field mapping corrected: {result}")
                return result
            return None

        except Exception as e:
            logger.warning(f"Field mapping correction error: {e}")
            return None
