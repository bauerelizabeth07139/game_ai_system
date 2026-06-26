<div align="center">

# 🎮 Game AI Control System

**通用游戏AI控制系统 — 读取屏幕截图，自动输出键鼠操作**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

<br/>

**感知 → 决策 → 策略 → 执行** 四层实时架构

10 FPS 主循环 · LLM 战术决策 · MLP 策略网络 · PID 速度控制

[快速开始](#-快速开始) · [架构详解](#-系统架构) · [配置说明](#-配置文件) · [扩展指南](#-扩展指南)

</div>

---

## ✨ 特性

- 🧠 **四层架构** — 感知层（HSV + OCR）→ 决策层（LLM API）→ 策略层（MLP）→ 执行层（PID + pynput）
- 🎯 **多游戏类型** — 内置 FPS / 格斗 / MOBA / 开放世界 四种专属状态空间
- 🤖 **多模型支持** — OpenAI / DeepSeek / 通义千问 / 智谱GLM / Ollama / vLLM 一键切换
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
├── requirements.txt           # Python 依赖
├── test_all.py                # 全量功能测试（12项）
│
├── install.bat / install.ps1  # 一键安装
├── run.bat / run.ps1          # 一键运行（日志模式）
├── run_gui.bat / run_gui.ps1  # 一键运行（GUI模式）
├── run_real.bat / run_real.ps1# 一键运行（真实键鼠）
├── build.bat / build.ps1      # PyInstaller 打包
│
└── src/
    ├── perception/            # 感知层
    │   ├── color_engine.py    #   HSV 颜色视觉引擎
    │   ├── ocr_engine.py      #   PaddleOCR 数值提取
    │   ├── time_buffer.py     #   环形缓冲区（50帧）
    │   ├── semantic_labeler.py#   数值 → 自然语言标签
    │   └── state_mapper.py    #   状态标准化映射
    │
    ├── decision/              # 决策层
    │   ├── llm_client.py      #   HTTP 调用大模型
    │   └── prompt_builder.py  #   提示词构建器
    │
    ├── strategy/              # 策略层
    │   ├── mlp_model.py       #   3层 MLP 网络定义
    │   └── inference.py       #   推理引擎
    │
    └── execution/             # 执行层
        ├── pid_controller.py  #   PID 速度控制器
        ├── noise_injector.py  #   正弦波抖动注入
        └── hal.py             #   硬件抽象层
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
