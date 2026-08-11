<div align="center">

# 🎮 Game AI Control System

**通用游戏AI控制系统 — 读取屏幕截图，自动输出键鼠操作**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()
[![Minecraft](https://img.shields.io/badge/Minecraft-AI%20Mod-brightgreen?logo=minecraft&logoColor=white)](https://www.minecraft.net/)
[![Fabric](https://img.shields.io/badge/Fabric-1.20.4-blue?logo=fabric&logoColor=white)](https://fabricmc.net/)
[![PPO](https://img.shields.io/badge/RL-PPO-red)](https://arxiv.org/abs/1707.06347)

<br/>

**感知 → 决策 → 策略 → 执行** 四层实时架构 · **Minecraft AI 玩家**

10 FPS 主循环 · LLM 战术决策 · MLP 策略网络 · PID 速度控制 · **PPO 强化学习**

[快速开始](#-快速开始) · [Minecraft AI Mod](#-minecraft-ai-mod-) · [架构详解](#-系统架构) · [训练指南](#-ai-训练指南) · [配置说明](#-配置文件) · [扩展指南](#-扩展指南)

</div>

---

## ✨ 特性

- 🧠 **四层架构** — 感知层（HSV + OCR）→ 决策层（LLM API）→ 策略层（MLP）→ 执行层（PID + pynput）
- 🎯 **多游戏类型** — 内置 FPS / 格斗 / MOBA / 开放世界 四种专属状态空间
- 🤖 **Minecraft AI 玩家** — 完整的 Fabric Mod + Python AI 大脑，自主通关 Minecraft
- 🏆 **PPO 强化学习** — 课程学习 → 从砍树到击败末影龙，分阶段训练

---
## 🎮 Minecraft AI Mod (NEW!)

**完整的 Minecraft AI 玩家系统** — 在游戏中创建一个由 AI 自主控制的玩家实体，具备所有正常玩家能力，目标是通关 Minecraft（击败末影龙）。

### 系统组成

```
minecraft-ai-mod/
├── 📦 Fabric Mod (Java 1.20.4)     ← 在 Minecraft 服务器端运行
│   ├── AIPlayer.java               ← AI 玩家实体（生存模式，全能力）
│   ├── GoalPlanner.java            ← 17 阶段目标规划器
│   ├── CombatController.java       ← 自动战斗系统
│   ├── InventoryManager.java       ← 物品管理 & 装备
│   ├── AITCPServer.java            ← TCP 服务器（接收 AI 指令）
│   └── ChunkLoading                ← 自动加载 AI 玩家周围区块
│
├── 🧠 Python AI Brain              ← 独立运行的 AI 决策系统
│   ├── mc_perception.py            ← 世界状态 → 141 维状态向量
│   ├── mc_policy.py                ← MLP 策略网络（456→512→1024→512→40）
│   ├── mc_controller.py            ← TCP 客户端 & 指令翻译器
│   ├── mc_environment.py           ← RL 环境（Gym 接口）
│   ├── mc_goals.py                 ← 17 阶段目标规划 + 进程序列
│   └── mc_prompt.py                ← LLM 提示词构建器
│
├── 🤖 Bot Client                   ← 连接 Mod 并运行 AI
│   ├── mc_bot.py                   ← 主 AI 机器人（10 FPS 主循环）
│   └── bot_launcher.py             ← 命令行启动器
│
└── 🏋️ Training Infrastructure      ← PPO + 课程学习训练
    ├── ppo_agent.py                ← PPO 算法（Actor-Critic）
    ├── minecraft_env.py            ← 模拟训练环境
    └── train.py                    ← 训练脚本（5300 episodes）
```

### 通关路线图 (Glass Cannon Speedrun)

```
砍树 → 制作工具 → 挖石 → 找食物 → 挖铁 → 制作盔甲
    → 挖钻石 → 找岩浆 → 建造传送门 → 进入地狱
    → 寻找堡垒 → 击杀烈焰神 → 寻找末地传送门
    → 进入末地 → 击杀末影龙 🎉
```

### 快速启动

**1. 安装 Fabric Mod**
```bash
cd minecraft_ai_mod
# 需要 JDK 17+ 和 Gradle
./gradlew build        # Linux/macOS
gradlew.bat build      # Windows
# 将 build/libs/minecraft-ai-mod-1.0.0.jar 放入服务器的 mods/ 目录
```

**2. 启动 AI 机器人**
```bash
# 确保 Minecraft 服务器（带 AI Mod）已在运行
python bot/bot_launcher.py --host localhost --port 25575

# 可选：启用 LLM 决策
python bot/bot_launcher.py --llm --fps 15

# 可选：使用预训练策略模型
python bot/bot_launcher.py --policy checkpoints/pretrained_policy.pt

# 指定初始目标
python bot/bot_launcher.py --goal GATHER_WOOD
```

**3. 游戏内命令**
```
/aiplayer spawn     ← 生成 AI 玩家
/aiplayer start     ← 激活 AI
/aiplayer stop      ← 暂停 AI
/aiplayer status    ← 查看 AI 状态
/aiplayer teleport  ← 将 AI 传送到你身边
```

### AI 能力矩阵

| 能力 | 实现方式 | 说明 |
|------|---------|------|
| 移动 | MLP 策略 → WASD / Sprint / Jump | 路径规划 + 碰撞检测 |
| 视角控制 | MLP 策略 → 鼠标速度 → 偏航/俯仰 | PID 平滑控制 |
| 挖掘方块 | 自动选择最佳工具 + 持续攻击 | GoalPlanner 驱动 |
| 放置方块 | BlockItem.place() | 建筑和传送门 |
| 战斗 | CombatController → 自动攻击 + 闪避 | 8 格检测范围 |
| 合成物品 | 简易合成系统（原木→木板→工具...） | 自动材料收集 |
| 物品管理 | InventoryManager → 自动整理 + 择优装备 | 丢弃垃圾物品 |
| 进食 | 饥饿度 < 15 自动进食 | 食物检测 |
| 区块加载 | ChunkTicketType → 4 格半径强制加载 | 保证 AI 周围环境活跃 |

### AI 决策流程

```
世界状态 (JSON/TCP)
    ↓
MCPerception → 141 维状态向量
    ↓
TimeBuffer (3 时间点历史) + 策略嵌入 (32 维) + 游戏类型 (1 维)
    ↓
MCPolicy (456→512→1024→512→40 MLP)
    ↓
40 维动作向量 (Sigmoid [0,1])
    ↓
MCActionSpace.to_minecraft_command()
    ↓
TCP → Fabric Mod → Minecraft 执行
```

**可选 LLM 决策路径**：每 50 帧通过大模型分析当前状态，输出高层战术决策（EXPLORE/MINE/ATTACK/CRAFT 等）。

---
- 🖥️ **可视化 GUI** — 实时显示战况、键盘热力图、鼠标轨迹、策略决策
- ⚡ **轻量推理** — MLP 策略网络 < 1ms 推理，不阻塞主循环
- 🔧 **零代码配置** — 所有参数通过 `config.yaml` 调整
- 📦 **一键部署** — `.bat` / `.ps1` 脚本双击即用

---

## 🚀 快速开始

### 方式一：双击脚本（推荐）

```
1. 双击 install.bat     ← 一键安装依赖
2. 双击 run.bat         ← 启动（日志模式）
3. 双击 run_gui.bat     ← 启动（可视化界面）
4. 双击 run_real.bat    ← 启动（真实键鼠输出）
```

### 方式二：命令行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动系统
python main.py

# 带 GUI 启动
python main.py --gui

# 真实键鼠模式（⚠️ 会控制你的鼠标键盘）
python main.py --real

# 限制运行帧数（测试用）
python main.py --frames 30 --fps 10
```

### 方式三：PowerShell

```powershell
.\install.ps1       # 安装
.\run.ps1           # 日志模式
.\run_gui.ps1       # GUI 模式
.\run_real.ps1      # 真实键鼠
.\test.bat          # 运行测试
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          主循环 (10 FPS)                            │
│                      每帧 100ms，截屏 → 处理 → 输出                  │
├──────────────┬──────────────┬───────────────┬───────────────────────┤
│   👁️ 感知层   │   🧠 决策层   │   ⚡ 策略层    │   🎮 执行层           │
│  Perception  │  Decision    │   Strategy    │   Execution           │
│              │              │               │                       │
│ HSV 颜色检测  │ LLM API 调用  │ MLP 神经网络   │ PID 速度控制           │
│ OCR 文字识别  │ 每 50 帧调用   │ 672→10 维     │ 噪声抖动注入           │
│ 语义标签生成  │ 异步非阻塞     │ 指令嵌入      │ pynput 键鼠模拟        │
│ 环形缓冲区   │ 规则降级兜底   │ < 1ms 推理    │ 时序队列               │
│ 状态映射器   │              │               │                       │
└──────────────┴──────────────┴───────────────┴───────────────────────┘
```

### 数据流

```
截图 → [感知层] → 141维状态向量 + 语义JSON
                         ↓                    ↓
                   [策略层 MLP]          [决策层 LLM] (每50帧)
                         ↓                    ↓
                   10维控制原语  ←──  策略指令 (FOCUS_FIRE/RETREAT/...)
                         ↓
                   [执行层] → PID平滑 → 噪声注入 → 键鼠事件
```

---

## 📁 项目结构

```
game_ai_system/
├── main.py                    # 主循环入口
├── gui.py                     # Tkinter 可视化界面
├── config.yaml                # 全部可调参数
├── mc_config.yaml             # Minecraft AI 配置
├── requirements.txt           # Python 依赖
├── test_all.py                # 全量功能测试（12项）
├── test_minecraft.py          # Minecraft AI 测试（19项）
│
├── install.bat / install.ps1  # 一键安装
├── run.bat / run.ps1          # 一键运行（日志模式）
├── run_gui.bat / run_gui.ps1  # 一键运行（GUI模式）
├── run_real.bat / run_real.ps1# 一键运行（真实键鼠）
├── build.bat / build.ps1      # PyInstaller 打包
│
├── minecraft_ai_mod/          # 📦 Fabric Mod (Java 1.20.4)
│   ├── build.gradle           #   Gradle 构建脚本
│   ├── src/main/java/         #   Java 源码
│   │   └── com/gameaisystem/minecraftmod/
│   │       ├── MinecraftAIMod.java     # 主 Mod 入口
│   │       ├── AIPlayer.java           # AI 玩家实体
│   │       ├── GoalPlanner.java        # 17 阶段目标规划
│   │       ├── CombatController.java   # 自动战斗
│   │       ├── InventoryManager.java   # 物品管理
│   │       ├── AITCPServer.java        # TCP 通信
│   │       └── mixin/                  # Mixin 注入
│   └── src/main/resources/    #   资源文件
│
├── src/
│   ├── perception/            # 感知层
│   │   ├── color_engine.py    #   HSV 颜色视觉引擎
│   │   ├── ocr_engine.py      #   PaddleOCR 数值提取
│   │   ├── time_buffer.py     #   环形缓冲区（50帧）
│   │   ├── semantic_labeler.py#   数值 → 自然语言标签
│   │   └── state_mapper.py    #   状态标准化映射
│   │
│   ├── decision/              # 决策层
│   │   ├── llm_client.py      #   HTTP 调用大模型
│   │   └── prompt_builder.py  #   提示词构建器
│   │
│   ├── strategy/              # 策略层
│   │   ├── mlp_model.py       #   3层 MLP 网络定义
│   │   └── inference.py       #   推理引擎
│   │
│   ├── execution/             # 执行层
│   │   ├── pid_controller.py  #   PID 速度控制器
│   │   ├── noise_injector.py  #   正弦波抖动注入
│   │   └── hal.py             #   硬件抽象层
│   │
│   └── minecraft/             # 🧠 Minecraft AI Brain
│       ├── mc_perception.py   #   世界状态 → 141 维向量
│       ├── mc_policy.py       #   MLP 策略网络 (456→40)
│       ├── mc_controller.py   #   TCP 客户端 & 指令翻译
│       ├── mc_environment.py  #   RL 环境 (Gym 接口)
│       ├── mc_goals.py        #   17 阶段目标规划
│       └── mc_prompt.py       #   LLM 提示词构建
│
├── bot/                       # 🤖 Bot Client
│   ├── mc_bot.py              #   主 AI 机器人 (10 FPS)
│   └── bot_launcher.py        #   命令行启动器
│
├── training/                  # 🏋️ Training Infrastructure
│   ├── ppo_agent.py           #   PPO 算法 (Actor-Critic)
│   ├── minecraft_env.py       #   模拟训练环境
│   └── train.py               #   训练脚本
│
└── checkpoints/               # 📊 Pre-trained Models
    └── pretrained_policy.pt   #   预训练策略网络
```

---

## 🔧 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | config.yaml | 配置文件路径 |
| `--gui` | flag | false | 启用可视化界面 |
| `--real` | flag | false | 启用真实键鼠输出 |
| `--fps` | int | 10 | 帧率 |
| `--frames` | int | 0 | 运行帧数限制（0=无限） |
| `--log-level` | str | INFO | 日志级别：DEBUG/INFO/WARNING/ERROR |

**组合示例：**

```bash
python main.py --gui --fps 15 --log-level DEBUG
python main.py --real --frames 500 --config my_config.yaml
```

---

## 🧩 四层架构详解

### 👁️ 感知层 (Perception)

从游戏截图提取结构化信息。

**ColorEngine — 颜色视觉引擎**

```
BGR截图 → 等比缩放(短边480px) → HSV颜色空间 → 双阈值掩膜 → 轮廓提取 → 几何筛选
```

- 面积筛选：50~2000 像素
- 长宽比：2:1 ~ 6:1
- 位置：仅屏幕上半部分（Y < 60%）
- 血量计算：`红色像素列宽 / (红色+灰色)列宽`

**SemanticLabeler — 语义标签**

| 数值范围 | 标签 | 威胁等级 |
|---------|------|---------|
| hp > 70% | 健康 | — |
| 30% ≤ hp ≤ 70% | 受伤 | — |
| hp < 30% | 残血(斩杀线) | +近距离 → 高威胁-集火目标 |

**TimeBuffer — 环形缓冲区**

| 接口 | 帧偏移 | 输出维度 | 用途 |
|------|--------|---------|------|
| `sample_for_mlp()` | [0, -1, -2, -4, -8] | 141×5 = 705 | 策略层输入 |
| `sample_for_decision()` | [0, -4, -12, -24, -40] | 5帧语义数据 | 决策层历史 |

### 🧠 决策层 (Decision)

通过 HTTP API 调用外部大模型，**每 50 帧（5秒）** 调用一次，异步不阻塞。

**支持的模型提供方：**

| 提供方 | api_type | 说明 |
|--------|----------|------|
| OpenAI | openai | GPT-4o / GPT-4o-mini |
| DeepSeek | deepseek | deepseek-chat |
| 通义千问 | qwen | qwen-turbo / qwen-plus |
| 智谱GLM | zhipu | glm-4 |
| Ollama | ollama | 本地模型 |
| vLLM | vllm | 本地部署 |

**决策输出格式：**

```json
{"strategy": "FOCUS_FIRE", "target": "E1", "priority": "high", "reasoning": "敌方E1残血，优先集火"}
```

**可用策略：** `FOCUS_FIRE` · `RETREAT` · `FLANK` · `DEFEND` · `PUSH` · `HOLD` · `ROTATE` · `ENGAGE` · `DISENGAGE`

> 💡 API 不可用时自动降级为内置规则决策器

### ⚡ 策略层 (Strategy)

本地 MLP 网络，将战略指令转为控制原语。

```
输入: 456维 (141×3状态 + 32维指令嵌入 + 1维时间)
  ↓
隐藏层1: 512维, ReLU
  ↓
隐藏层2: 256维, ReLU
  ↓
输出: 40维 (4连续 + 36按键)
```

**输出维度：**

| 维度 | 含义 | 激活函数 | 范围 |
|------|------|---------|------|
| 0-1 | 鼠标速度 (vx, vy) | Tanh | [-1, 1] |
| 2-3 | 移动速度 (MoveX, MoveY) | Tanh | [-1, 1] |
| 4-39 | 按键触发概率 × 36 | Sigmoid | [0, 1] |

### 🎮 执行层 (Execution)

**PIDController** — 速度平滑控制

```
目标速度 [-1,1] → PID(Kp=1.2, Ki=0.01, Kd=0.1) → 平滑速度 → 位移
每帧位移 = 速度 × 灵敏度 × 帧间隔 = 0.5 × 100 × 0.1 = 5.0 px/frame
```

**NoiseInjector** — 模拟人类操作

```python
noise = 0.02 × sin(2π × 8.0 × t)  # 8Hz 正弦波，振幅 0.02
```

---

## ⚙️ 配置文件

所有参数通过 `config.yaml` 调整，无需修改代码：

```yaml
# 全局参数
global:
  fps: 10                    # 系统帧率
  short_side: 480            # 截图缩放短边
  buffer_size: 50            # 缓冲区大小
  log_level: INFO
  window_title: ""           # 游戏窗口标题（空=全屏）

# 感知层
perception:
  hsv:
    red_lower_1: [0, 50, 50]     # 红色（敌人）
    red_upper_1: [10, 255, 255]
    green_lower: [35, 50, 50]    # 绿色（队友）
    green_upper: [85, 255, 255]
  ocr:
    enabled: true

# 决策层
decision:
  api_type: openai
  api_base: https://api.deepseek.com/v1
  api_key: "sk-xxx"
  model_name: deepseek-chat
  interval_frames: 50

# 策略层
strategy:
  state_dim: 141
  embed_dim: 32

# 执行层
execution:
  pid: {Kp: 1.2, Ki: 0.01, Kd: 0.1}
  noise: {frequency: 8.0, amplitude: 0.02}
  mouse_sensitivity: 100
```

---

## 🖥️ 可视化界面

```bash
python main.py --gui
```

GUI 包含：
- 📊 实时战况面板（敌我数量、血量、策略）
- ⌨️ 键盘热力图（实时显示触发按键）
- 🖱️ 鼠标轨迹可视化
- 📝 决策日志滚动显示
- ⚙️ 运行时参数调整面板

---

## 🧪 测试

```bash
# 运行全量测试（12项）
python test_all.py

# 主循环测试（20帧）
python main.py --frames 20 --fps 10
```

| 测试项 | 说明 |
|--------|------|
| ColorEngine | HSV 颜色检测 |
| SemanticLabeler | 语义标签生成 |
| TimeBuffer | 环形缓冲区 |
| OCREngine | OCR 禁用模式 |
| LLMClient | 模拟决策 |
| PromptBuilder | 提示词构建 |
| MLPModel | 前向传播 |
| StrategyInference | 推理引擎 |
| PIDController | 速度控制 |
| NoiseInjector | 噪声注入 |
| HAL | 整合测试 |
| 完整流水线 | 10帧端到端 |

---

## 📊 性能指标

| 层 | 耗时 | 说明 |
|----|------|------|
| 感知层（不含OCR） | < 10ms | HSV + 轮廓提取 |
| 感知层（含OCR） | < 100ms | + PaddleOCR |
| 策略层 | < 1ms | MLP 前向传播 |
| 决策层 | 异步 | 不阻塞主循环 |
| 执行层 | < 1ms | PID + 事件生成 |

---

## 🏋️ AI 训练指南

### 课程学习体系

AI 通过 **PPO (Proximal Policy Optimization)** 强化学习算法，分 6 个阶段逐步训练：

| 阶段 | 目标 | episodes | 奖励阈值 | 说明 |
|------|------|----------|---------|------|
| 1 | GATHER_WOOD | 500 | 10.0 | 学会砍树收集原木 |
| 2 | CRAFT_TOOLS | 500 | 15.0 | 制作木板、工具 |
| 3 | MINE_STONE | 500 | 20.0 | 挖掘石制工具 |
| 4 | MINE_IRON | 800 | 25.0 | 地下挖铁矿 |
| 5 | MINE_DIAMOND | 1000 | 30.0 | Y=-54 深挖钻石 |
| 6 | BEAT_DRAGON | 2000 | 100.0 | 击败末影龙 |

### 奖励设计

```
基础存活奖励：+0.01/步
生命值下降惩罚（无战斗）：-2.0×hp_change
生命值下降惩罚（战斗中）：-0.1×hp_change
经验值奖励：+5.0×xp_gained
物品收集奖励：+2.0×new_items
饥饿惩罚（hunger<5）：-1.0
战斗奖励（击败敌人）：+1.0/敌人
```

### 里程碑奖金

| 事件 | 奖励 | 自动切换目标 |
|------|------|-------------|
| 获取第一个原木 | +5.0 | → CRAFT_TOOLS |
| 制作石制工具 | +10.0 | → MINE_IRON |
| 获得铁锭 | +15.0 | → MINE_DIAMOND |
| 获得钻石 | +20.0 | → FIND_LAVA |
| 进入地狱 | +30.0 | → FIND_FORTRESS |
| 获得烈焰棒 | +15.0 | → FIND_END_PORTAL |
| 进入末地 | +50.0 | → KILL_ENDER_DRAGON |
| 击败末影龙 | +100.0 | → DONE! |

### 开始训练

```bash
# 从头训练
python training/train.py --episodes 5000 --save-dir ./checkpoints

# 断点续训
python training/train.py --load checkpoints/checkpoint_1000.pt

# GPU 加速训练
python training/train.py --device cuda --batch-size 128

# 评估模式（不训练）
python training/train.py --evaluate --load checkpoints/best_model.pt
```

**训练日志**：自动保存至 `training.log`，每 10 episodes 输出统计信息。

### 输出文件

```
checkpoints/
├── checkpoint_100.pt / policy_100.pt    ← 每 100 episodes
├── checkpoint_200.pt / policy_200.pt
├── ...
├── best_model.pt / best_policy.pt       ← 最佳模型
├── final_model.pt / final_policy.pt     ← 最终模型
└── pretrained_policy.pt                 ← 预训练基线
```

---

## 🔌 扩展指南

### 接入真实游戏截屏

```python
# main.py → _capture_screen()
import mss
with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)[:, :, :3]
```

### 调整 HSV 阈值

```python
import cv2, numpy as np
hsv = cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2HSV)
print(hsv)  # 获取 H, S, V 值 → 写入 config.yaml
```

### 接入自定义 LLM

```yaml
decision:
  api_type: openai
  api_base: https://your-api.com/v1
  api_key: "your-key"
  model_name: your-model
```

### 添加新游戏类型

1. 在 `src/strategy/mlp_model.py` 的 `GAME_TYPE_STATE_RANGES` 添加状态区间
2. 在 `main.py` 的 `_build_state_vector()` 添加专属特征
3. 在 `config.yaml` 设置 `decision.game_type`

---

## 📋 环境要求

- **Python** 3.10+
- **操作系统** Windows / Linux / macOS
- **GPU** 可选（CPU 推理即可，MLP < 1ms）

### 依赖清单

| 包 | 版本 | 用途 | 必需 |
|----|------|------|------|
| opencv-python | ≥4.8.0 | HSV 检测、图像处理 | ✅ |
| numpy | ≥1.24.0 | 数值计算 | ✅ |
| torch | ≥2.0.0 | MLP 策略网络 | ✅ |
| pyyaml | ≥6.0 | 配置文件解析 | ✅ |
| requests | ≥2.31.0 | HTTP API 调用 | ✅ |
| pynput | ≥1.7.6 | 键鼠模拟 | ✅ |
| mss | ≥9.0.0 | 屏幕截图 | ✅ |
| paddlepaddle | ≥2.5.0 | OCR 引擎 | ❌ 可选 |
| paddleocr | ≥2.7.0 | PP-OCRv6 文字识别 | ❌ 可选 |

---

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，给个 Star 吧！**

</div>
