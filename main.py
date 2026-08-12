"""游戏AI控制系统 - 主循环

以10FPS固定频率运行：
- 每帧运行感知层并存入缓冲区
- 每帧运行策略层得到控制原语
- 决策层每50帧异步调用一次API
"""
import os
import sys
import time
import logging
import argparse
import signal
import threading
import cv2
import numpy as np
import yaml

# 修复Windows控制台中文乱码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 项目路径（兼容PyInstaller打包）
if getattr(sys, 'frozen', False):
    # PyInstaller打包后，exe所在目录
    PROJECT_DIR = os.path.dirname(sys.executable)
    # 临时解压目录（用于导入模块）
    _bundle_dir = sys._MEIPASS
    sys.path.insert(0, _bundle_dir)
else:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, PROJECT_DIR)

from src.perception.color_engine import ColorEngine
from src.perception.ocr_engine import OCREngine
from src.perception.time_buffer import TimeBuffer
from src.perception.semantic_labeler import SemanticLabeler
from src.perception.state_mapper import StateMapper
from src.decision.llm_client import LLMClient
from src.decision.prompt_builder import PromptBuilder
from src.strategy.inference import StrategyInference
from src.execution.hal import HAL


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(PROJECT_DIR, 'config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return {}
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {}


def setup_logging(level: str = 'INFO'):
    """配置日志"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )


class GameAISystem:
    """游戏AI控制系统主类"""
    
    def __init__(self, config: dict, real_input: bool = False, max_frames: int = 0):
        self.config = config
        self.logger = logging.getLogger('GameAI')
        self._max_frames = max_frames  # 0=无限
        
        # 全局参数
        global_cfg = config.get('global', {})
        self.fps = global_cfg.get('fps', 10)
        self.frame_interval = 1.0 / self.fps
        self.short_side = global_cfg.get('short_side', 480)
        
        # 初始化各层
        self.logger.info("初始化感知层...")
        self.color_engine = ColorEngine(config.get('perception', {}))
        self.ocr_engine = OCREngine(config.get('perception', {}).get('ocr', {}))
        self.time_buffer = TimeBuffer(config.get('buffer', {}))
        self.semantic_labeler = SemanticLabeler(config.get('perception', {}).get('semantic', {}))
        self.state_mapper = StateMapper(config.get('perception', {}).get('state_mapper', {
            'game_type': config.get('decision', {}).get('game_type', 'unknown'),
            'screen_width': 960,
            'screen_height': 540
        }))

        self.logger.info("初始化决策层...")
        self.llm_client = LLMClient(config.get('decision', {}))
        self.prompt_builder = PromptBuilder(config.get('decision', {}))
        
        self.logger.info("初始化策略层...")
        self.strategy = StrategyInference(config.get('strategy', {}))
        
        self.logger.info("初始化执行层...")
        self.hal = HAL(config.get('execution', {}), real_input=real_input)
        
        # 运行状态
        self._running = False
        self._frame_count = 0
        self._start_time = 0
        self._gui_root = None  # GUI窗口引用（用于截图时隐藏）
        
        # 模拟模式（API不可用时）
        self._simulate_on_failure = config.get('decision', {}).get('simulate_on_failure', True)
    
    def set_gui_root(self, root):
        """设置GUI窗口引用（截图时临时隐藏避免截到自己）"""
        self._gui_root = root
    
    def _capture_screen(self) -> np.ndarray:
        """捕获屏幕截图
        
        优先截取指定游戏窗口，找不到则全屏截图
        全屏截图时会临时隐藏GUI窗口避免截到自己
        """
        window_title = self.config.get('global', {}).get('window_title', '')
        
        # 尝试窗口截图
        if window_title:
            try:
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                
                # 查找窗口
                hwnd = user32.FindWindowW(None, window_title)
                if hwnd:
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
                    
                    import mss
                    with mss.MSS() as sct:
                        monitor = {"left": left, "top": top,
                                   "width": right - left, "height": bottom - top}
                        screenshot = sct.grab(monitor)
                        img = np.array(screenshot)
                        if img.shape[2] == 4:
                            img = img[:, :, :3]
                        return img
            except Exception:
                pass
        
        # 全屏截图
        try:
            import mss
            with mss.MSS() as sct:
                monitor = sct.monitors[1]  # 主显示器
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
            return img
        except ImportError:
            # mss不可用，生成模拟图像
            h, w = 540, 960
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            cv2.rectangle(img, (200, 100), (250, 120), (0, 0, 200), -1)
            cv2.rectangle(img, (600, 150), (640, 165), (0, 0, 180), -1)
            cv2.rectangle(img, (400, 200), (440, 215), (0, 200, 0), -1)
            return img
        except Exception as e:
            self.logger.error(f"屏幕捕获失败: {e}")
            return np.zeros((540, 960, 3), dtype=np.uint8)
    
    def _perception_step(self, frame_image: np.ndarray) -> dict:
        """感知层处理单帧"""
        # 颜色检测（process内部已做缩放）
        detection = self.color_engine.process(frame_image)

        # 缩放图像用于OCR和语义标签
        resized = self.color_engine.resize_image(frame_image, self.short_side)

        # OCR（可选，较慢）
        ocr_texts = []
        if self.ocr_engine.enabled:
            try:
                ocr_texts = self.ocr_engine.extract_text(resized)
            except Exception:
                pass

        # 生成语义数据
        semantic_data = self.semantic_labeler.generate_frame_data(
            frame_id=self._frame_count,
            enemies=detection.get('enemies', []),
            allies=detection.get('allies', []),
            image_size=resized.shape[:2][::-1]  # (w, h)
        )

        # 状态映射器：整理检测结果为标准化状态
        mapped_state = self.state_mapper.process(detection, semantic_data)

        # 构建状态向量（141维，包含通用+游戏专属）
        state_vector = self._build_state_vector(detection)

        # 存入缓冲区
        buffer_entry = {
            'frame_id': self._frame_count,
            'state_vector': state_vector,
            'semantic_data': semantic_data,
            'mapped_state': mapped_state,
            'notifications': self.state_mapper.get_notifications_text(),
            'detection': detection,
            'ocr_texts': ocr_texts
        }
        self.time_buffer.push(buffer_entry)

        return buffer_entry
    
    def _build_state_vector(self, detection: dict) -> np.ndarray:
        """构建141维状态向量（所有游戏类型专属特征全部拼接）

        141维构成：
        ├── 通用状态 [0-49] 50维
        │   [0-5]   自身: 血量,蓝量,坐标XY,朝向Sin/Cos
        │   [6-25]  5敌方: 相对XY,血量,存活 各4维
        │   [26-45] 5友方: 同上
        │   [46-49] 地图: 目标方位2维,资源血量,时间进度
        │
        ├── FPS专属 [50-74] 25维
        │   弹药,护甲,准心XY,经济,回合,技能CD×4,距离,视野,
        │   瞄准,移动,蹲跳,闪光/烟雾/手雷,拆弹,炸弹,声音,受伤,烟雾中
        │
        ├── 格斗专属 [75-96] 22维
        │   能量槽×2,晕值×2,距离,状态×2,帧优势,连段,角落,
        │   防反CD,超必杀CD,硬直,防御,对手硬直,空中×2,Burst,RISC,伤害,位置,必杀
        │
        ├── MOBA专属 [97-118] 22维
        │   等级,金币,经验,技能CD×4,补刀,击杀,死亡,助攻,
        │   血量%,法力%,召唤师CD×2,防御塔×2,龙CD,大龙CD,兵线,回城,视野
        │
        └── 开放世界专属 [119-140] 22维
            体力,饱食度,温度,元素,反应CD,采集CD,骑乘,坐骑体力,
            任务进度×2,战斗,Boss,资源距离,天气,昼夜,滑翔,游泳,攀爬,背包,武器,耐久,神瞳
        """
        sv = np.zeros(141, dtype=np.float32)

        # ═══ 通用状态 [0-49] ═══
        sv[0] = 0.8   # 血量
        sv[1] = 0.6   # 蓝量/能量
        sv[2] = 0.0   # X坐标（归一化）
        sv[3] = 0.0   # Y坐标（归一化）
        sv[4] = 0.0   # 朝向sin
        sv[5] = 1.0   # 朝向cos

        enemies = detection.get('enemies', [])
        for i in range(min(5, len(enemies))):
            e = enemies[i]
            cx, cy = e.get('center', (0, 0))
            base = 6 + i * 4
            sv[base] = cx / 480.0 - 0.5
            sv[base + 1] = cy / 270.0 - 0.5
            sv[base + 2] = e.get('health', 0.0)
            sv[base + 3] = 1.0

        allies = detection.get('allies', [])
        for i in range(min(5, len(allies))):
            a = allies[i]
            cx, cy = a.get('center', (0, 0))
            base = 26 + i * 4
            sv[base] = cx / 480.0 - 0.5
            sv[base + 1] = cy / 270.0 - 0.5
            sv[base + 2] = a.get('health', 0.0)
            sv[base + 3] = 1.0

        sv[46] = 0.0   # 目标点方位X
        sv[47] = 0.0   # 目标点方位Y
        sv[48] = 1.0   # 关键资源血量
        sv[49] = 0.5   # 时间进度

        # ═══ FPS专属 [50-74] 25维 ═══
        sv[50] = 0.8   # 弹药量
        sv[51] = 0.5   # 护甲值
        sv[52] = 0.0   # 准心X
        sv[53] = 0.0   # 准心Y
        sv[54] = 0.6   # 经济/金钱
        sv[55] = 0.3   # 回合进度
        sv[56] = 0.0   # 技能1 CD
        sv[57] = 0.0   # 技能2 CD
        sv[58] = 0.0   # 技能3 CD
        sv[59] = 0.0   # 技能4/大招 CD
        sv[60] = 0.5   # 最近敌人距离
        sv[61] = 0.3   # 视野内敌人数量
        sv[62] = 0.0   # 是否在瞄准
        sv[63] = 1.0   # 是否在移动
        sv[64] = 0.0   # 是否蹲下
        sv[65] = 0.0   # 是否跳跃
        sv[66] = 0.5   # 弹药携带量
        sv[67] = 0.0   # 闪光弹数量
        sv[68] = 0.0   # 烟雾弹数量
        sv[69] = 0.0   # 手雷数量
        sv[70] = 0.0   # 拆弹进度
        sv[71] = 0.0   # 是否持有炸弹
        sv[72] = 0.3   # 声音方位
        sv[73] = 0.0   # 受到伤害方向
        sv[74] = 0.0   # 是否在烟雾中

        # ═══ 格斗专属 [75-96] 22维 ═══
        sv[75] = 0.5   # 自身能量槽
        sv[76] = 0.3   # 对手能量槽
        sv[77] = 0.0   # 自身晕值
        sv[78] = 0.0   # 对手晕值
        sv[79] = 0.5   # 距离
        sv[80] = 0.0   # 对手状态
        sv[81] = 0.0   # 自身状态
        sv[82] = 0.0   # 帧优势
        sv[83] = 0.0   # 连段计数
        sv[84] = 0.0   # 是否在角落
        sv[85] = 0.0   # 防反技能CD
        sv[86] = 0.0   # 超必杀CD
        sv[87] = 0.0   # 是否在硬直中
        sv[88] = 0.0   # 是否在防御
        sv[89] = 0.0   # 对手是否在硬直
        sv[90] = 0.0   # 是否在空中
        sv[91] = 0.0   # 对手是否在空中
        sv[92] = 0.0   # Burst槽
        sv[93] = 0.0   # RISC槽
        sv[94] = 0.0   # 连段伤害
        sv[95] = 0.5   # 屏幕位置
        sv[96] = 0.0   # 是否正使用必杀技

        # ═══ MOBA专属 [97-118] 22维 ═══
        sv[97] = 0.3   # 等级
        sv[98] = 0.4   # 金币
        sv[99] = 0.3   # 经验
        sv[100] = 0.0  # Q技能 CD
        sv[101] = 0.0  # W技能 CD
        sv[102] = 0.0  # E技能 CD
        sv[103] = 0.0  # R技能/大招 CD
        sv[104] = 0.5  # 补刀数
        sv[105] = 0.0  # 击杀数
        sv[106] = 0.0  # 死亡数
        sv[107] = 0.0  # 助攻数
        sv[108] = 0.5  # 当前生命值百分比
        sv[109] = 0.5  # 当前法力值百分比
        sv[110] = 0.0  # 召唤师技能1 CD (闪现)
        sv[111] = 0.0  # 召唤师技能2 CD (点燃/治疗)
        sv[112] = 0.5  # 防御塔血量
        sv[113] = 0.5  # 敌方防御塔血量
        sv[114] = 0.0  # 小龙刷新倒计时
        sv[115] = 0.0  # 大龙刷新倒计时
        sv[116] = 0.5  # 兵线位置
        sv[117] = 0.0  # 是否回城中
        sv[118] = 0.0  # 视野分数

        # ═══ 开放世界专属 [119-140] 22维 ═══
        sv[119] = 0.8  # 体力
        sv[120] = 0.6  # 饱食度
        sv[121] = 0.5  # 温度
        sv[122] = 0.0  # 附着元素
        sv[123] = 0.0  # 元素反应CD
        sv[124] = 0.0  # 采集技能CD
        sv[125] = 0.0  # 是否骑乘
        sv[126] = 0.5  # 坐骑体力
        sv[127] = 0.3  # 主线任务进度
        sv[128] = 0.1  # 支线任务进度
        sv[129] = 0.0  # 是否在战斗状态
        sv[130] = 0.0  # Boss阶段
        sv[131] = 0.5  # 最近资源点距离
        sv[132] = 0.0  # 天气
        sv[133] = 0.5  # 昼夜
        sv[134] = 0.0  # 滑翔中
        sv[135] = 0.0  # 游泳中
        sv[136] = 0.0  # 攀爬中
        sv[137] = 0.5  # 背包容量
        sv[138] = 0.0  # 当前武器类型
        sv[139] = 0.0  # 武器耐久
        sv[140] = 0.0  # 神瞳收集进度

        # ═══ 规则过滤器：只保留当前游戏类型的专属状态 ═══
        from src.strategy.mlp_model import GAME_TYPE_STATE_RANGES
        game_type = self.strategy.current_game_type if hasattr(self, 'strategy') else 'unknown'

        # unknown类型：只使用通用状态，所有专属状态清零
        if game_type == 'unknown':
            sv[50:141] = 0.0
        else:
            # 非unknown类型：只保留当前类型的专属状态，其他清零
            for gt, (start, end) in GAME_TYPE_STATE_RANGES.items():
                if gt != game_type:
                    sv[start:end] = 0.0

        return sv
    
    def _decision_step(self):
        """决策层处理（每50帧调用一次）"""
        if not self.llm_client.should_call(self._frame_count):
            return

        # 获取历史数据用于构建提示词
        decision_frames = self.time_buffer.sample_for_decision()
        offsets = self.llm_client.history_frames

        history_text = self.semantic_labeler.generate_history_text(decision_frames, offsets)

        # 获取当前语义数据
        current = self.time_buffer.get_latest()
        if not current:
            return

        semantic_data = current.get('semantic_data', {})
        notifications = current.get('notifications', 'No notifications')

        # 构建用户消息（包含通知）
        user_message = self.prompt_builder.build_from_semantic_data(semantic_data, history_text)

        # 添加通知到用户消息
        user_message = f"{user_message}\n\nNotifications:\n{notifications}"

        # 异步调用API
        self.llm_client.call_async(user_message, self._frame_count)
        
        # 更新策略
        decision = self.llm_client.get_decision()
        self.strategy.set_strategy(decision.get('strategy', 'HOLD'))
    
    def _strategy_step(self) -> np.ndarray:
        """策略层处理，返回控制原语"""
        # 从缓冲区获取5个时间点的状态向量
        state_vector = self.time_buffer.sample_for_mlp()
        
        # 推理
        action = self.strategy.infer(state_vector)
        
        return action
    
    def run(self):
        """运行主循环"""
        self.logger.info("=" * 60)
        self.logger.info("游戏AI控制系统启动")
        self.logger.info(f"帧率: {self.fps} FPS | 缩放: {self.short_side}px")
        self.logger.info(f"决策间隔: {self.llm_client.interval_frames}帧")
        self.logger.info(f"真实输入: {'启用' if self.hal.real_input else '禁用(仅日志)'}")
        self.logger.info("=" * 60)
        
        self._running = True
        self._start_time = time.perf_counter()
        self.hal.start()
        
        # 注册信号处理（仅主线程）
        try:
            signal.signal(signal.SIGINT, lambda s, f: self.stop())
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        except ValueError:
            pass  # 非主线程中无法注册信号
        
        try:
            while self._running:
                frame_start = time.perf_counter()
                
                # 1. 截屏
                frame_image = self._capture_screen()
                
                # 2. 感知层处理
                perception = self._perception_step(frame_image)
                
                # 3. 决策层（异步，每50帧）
                self._decision_step()
                
                # 4. 策略层推理
                action = self._strategy_step()
                
                # 5. 执行层
                self.hal.execute(action)
                
                # 6. 控制台输出
                self._print_status(perception, action)
                
                # 帧率控制
                elapsed = time.perf_counter() - frame_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self._frame_count += 1
                
                # 帧数限制检查
                if self._max_frames > 0 and self._frame_count >= self._max_frames:
                    self.logger.info(f"已达到帧数限制({self._max_frames})，停止运行")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
        finally:
            self.stop()
    
    def stop(self):
        """停止系统"""
        self._running = False
        self.hal.stop()
        self.logger.info("系统已停止")
    
    def _print_status(self, perception: dict, action: np.ndarray):
        """控制台输出状态"""
        semantic = perception.get('semantic_data', {})
        decision = self.llm_client.get_decision()
        
        enemies = semantic.get('enemies', [])
        allies = semantic.get('allies', [])

        action_summary = self.hal.format_action_summary(action)

        # 获取通知
        notifications = perception.get('notifications', '')
        notif_short = ''
        if 'CRIT' in notifications:
            notif_short = ' [!]'
        elif 'WARN' in notifications:
            notif_short = ' [*]'

        self.logger.info(
            f"帧{self._frame_count:4d} | "
            f"策略:{decision.get('strategy', 'HOLD'):10s} | "
            f"敌:{len(enemies)} 友:{len(allies)} | "
            f"{action_summary}{notif_short}"
        )


def main():
    parser = argparse.ArgumentParser(description='游戏AI控制系统')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--real', action='store_true', help='启用真实键鼠输入')
    parser.add_argument('--fps', type=int, default=None, help='覆盖帧率')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--frames', type=int, default=0, help='运行帧数限制（0=无限）')
    parser.add_argument('--gui', action='store_true', help='启用可视化界面')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖参数
    if args.fps:
        config.setdefault('global', {})['fps'] = args.fps
    
    # 设置日志
    setup_logging(args.log_level or config.get('global', {}).get('log_level', 'INFO'))
    
    if args.gui:
        _run_with_gui(config, args)
    else:
        # 创建并运行系统
        system = GameAISystem(config, real_input=args.real, max_frames=args.frames)
        system.run()


def _run_with_gui(config: dict, args):
    """带GUI模式运行"""
    from gui import GameAIGUI
    import threading

    gui = GameAIGUI()
    system = GameAISystem(config, real_input=args.real, max_frames=args.frames)
    system.set_gui_root(gui.root)  # 截图时隐藏GUI窗口

    # 设置回调：GUI配置变更时更新LLM客户端
    def on_settings_change(settings):
        system.llm_client.api_type = settings.get('api_type', 'openai')
        system.llm_client.api_base = settings.get('api_base', '')
        system.llm_client.api_key = settings.get('api_key', '')
        system.llm_client.model_name = settings.get('model_name', '')
        game_type = settings.get('game_type', 'auto')
        system.llm_client.set_game_type(game_type)
        system.strategy.set_game_type(game_type)
        # 如果api_base是提供方名称，自动替换
        from src.decision.llm_client import PROVIDER_URLS
        if system.llm_client.api_base in PROVIDER_URLS:
            system.llm_client.api_base = PROVIDER_URLS[system.llm_client.api_base]
        system.logger.info(f"配置更新: {system.llm_client.api_type}/{system.llm_client.model_name} game={game_type}")

    gui.set_settings_callback(on_settings_change)

    # 初始化GUI设置面板为当前配置值
    gui.set_settings({
        'api_type': system.llm_client.api_type,
        'api_base': system.llm_client.api_base,
        'api_key': system.llm_client.api_key,
        'model_name': system.llm_client.model_name,
        'game_type': system.llm_client.game_type,
    })

    # 重写系统的状态输出方法以更新GUI
    original_print = system._print_status

    def gui_print_status(perception, action):
        original_print(perception, action)
        semantic = perception.get('semantic_data', {})
        decision = system.llm_client.get_decision()
        enemies = semantic.get('enemies', [])
        allies = semantic.get('allies', [])

        # 计算PID平滑后的速度和位移
        # Sigmoid[0,1] -> [-1,1]
        target_vx = (action[0] - 0.5) * 2.0
        target_vy = (action[1] - 0.5) * 2.0
        smooth_v = system.hal.pid.update(np.array([target_vx, target_vy], dtype=np.float64))
        v_x, v_y = float(smooth_v[0]), float(smooth_v[1])
        dx = v_x * system.hal.mouse_sensitivity * system.hal.frame_interval
        dy = v_y * system.hal.mouse_sensitivity * system.hal.frame_interval

        # 移动速度 Sigmoid[0,1] -> [-1,1]
        move_vx = (action[2] - 0.5) * 2.0
        move_vy = (action[3] - 0.5) * 2.0

        button_probs = action[4:40]
        button_triggers = [bool(p > 0.5) for p in button_probs]

        gui.update_frame({
            'frame_id': system._frame_count,
            'fps': system.fps,
            'game_type': system.strategy.current_game_type,
            'strategy': decision.get('strategy', 'HOLD'),
            'target': decision.get('target', '--'),
            'priority': decision.get('priority', 'low'),
            'reasoning': decision.get('reasoning', ''),
            'enemies': [{'id': e.get('id', f'E{i+1}'), 'hp': e.get('hp', 0),
                         'hp_status': e.get('hp_status', ''), 'pos': e.get('pos', ''),
                         'tags': e.get('tags', ''), 'active': True}
                        for i, e in enumerate(enemies)],
            'allies': [{'id': a.get('id', f'A{i+1}'), 'hp': a.get('hp', 0),
                        'hp_status': a.get('hp_status', ''), 'pos': a.get('pos', '')}
                       for i, a in enumerate(allies)],
            'summary': semantic.get('summary', ''),
            'mouse_velocity': (target_vx, target_vy),
            'move_velocity': (move_vx, move_vy),
            'buttons': button_triggers,
            'pid_velocity': (v_x, v_y),
            'displacement': (dx, dy),
            'log_message': f"帧{system._frame_count}: {decision.get('strategy', 'HOLD')}",
        })

    system._print_status = gui_print_status

    # 在后台线程运行系统
    def run_system():
        system.run()
        # 系统停止后关闭GUI
        gui.root.after(500, gui.stop)

    thread = threading.Thread(target=run_system, daemon=True)
    thread.start()

    # 设置关闭回调
    def on_close():
        system.stop()
        gui.stop()

    gui.set_close_callback(on_close)

    # 启动GUI（主线程）
    gui.start()


if __name__ == '__main__':
    main()
