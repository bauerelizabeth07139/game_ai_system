"""游戏AI控制系统 - 实时可视化GUI"""
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

MAX_ENEMIES = 5
MAX_ALLIES = 5

# 40维按键名称（索引4-39）
KEY_NAMES = [
    'LMB', 'RMB', 'MMB',
    'W', 'A', 'S', 'D',
    'Space', 'Shift', 'Ctrl', 'Tab', 'Esc',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    'Q', 'E', 'R', 'F', 'G', 'H', 'Z', 'X', 'C', 'V',
    'B', 'N', 'T', 'M'
]

# GUI键盘布局（行, 列, 键名, 宽度占比）
KB_KEYS = [
    # (row, col, name, colspan)
    # 数字行
    (0, 0, 'Esc', 1), (0, 1, '1', 1), (0, 2, '2', 1), (0, 3, '3', 1),
    (0, 4, '4', 1), (0, 5, '5', 1), (0, 6, '6', 1), (0, 7, '7', 1),
    (0, 8, '8', 1), (0, 9, '9', 1), (0, 10, '0', 1),
    # QWERTY行
    (1, 0, 'Tab', 1), (1, 1, 'Q', 1), (1, 2, 'W', 1), (1, 3, 'E', 1),
    (1, 4, 'R', 1), (1, 5, 'T', 1), (1, 6, 'G', 1), (1, 7, 'H', 1),
    (1, 8, 'Z', 1), (1, 9, 'X', 1), (1, 10, 'C', 1),
    # ASDF行
    (2, 0, 'Ctrl', 1), (2, 1, 'A', 1), (2, 2, 'S', 1), (2, 3, 'D', 1),
    (2, 4, 'F', 1), (2, 5, 'V', 1), (2, 6, 'B', 1), (2, 7, 'N', 1),
    (2, 8, 'M', 1),
    # Shift + Space行
    (3, 0, 'Shift', 2), (3, 2, 'Space', 7),
    # 鼠标行
    (4, 0, 'LMB', 3), (4, 3, 'RMB', 3), (4, 6, 'MMB', 3),
]


class GameAIGUI:
    """实时可视化控制面板"""

    BG_DARK = "#1a1a2e"
    BG_PANEL = "#16213e"
    BG_CARD = "#0f3460"
    TEXT_PRIMARY = "#e6e6e6"
    TEXT_SECONDARY = "#a0a0a0"
    ACCENT_BLUE = "#4da6ff"
    ACCENT_GREEN = "#00e676"
    ACCENT_RED = "#ff5252"
    ACCENT_YELLOW = "#ffd740"
    ACCENT_ORANGE = "#ff9100"
    ACCENT_PURPLE = "#b388ff"
    PROGRESS_BG = "#2a2a4a"

    KEY_COLORS = {
        'LMB': ACCENT_RED, 'RMB': ACCENT_ORANGE, 'MMB': ACCENT_YELLOW,
        'W': ACCENT_GREEN, 'A': ACCENT_GREEN, 'S': ACCENT_GREEN, 'D': ACCENT_GREEN,
        'Space': ACCENT_BLUE, 'Shift': ACCENT_BLUE, 'Ctrl': ACCENT_BLUE,
        'Tab': ACCENT_BLUE, 'Esc': ACCENT_BLUE,
        'Q': ACCENT_PURPLE, 'E': ACCENT_PURPLE, 'R': ACCENT_PURPLE, 'F': ACCENT_PURPLE,
        'G': ACCENT_PURPLE, 'H': ACCENT_PURPLE, 'Z': ACCENT_PURPLE, 'X': ACCENT_PURPLE,
        'C': ACCENT_PURPLE, 'V': ACCENT_PURPLE,
        '1': TEXT_SECONDARY, '2': TEXT_SECONDARY, '3': TEXT_SECONDARY,
        '4': TEXT_SECONDARY, '5': TEXT_SECONDARY, '6': TEXT_SECONDARY,
        '7': TEXT_SECONDARY, '8': TEXT_SECONDARY, '9': TEXT_SECONDARY, '0': TEXT_SECONDARY,
        'B': TEXT_SECONDARY, 'N': TEXT_SECONDARY, 'T': TEXT_SECONDARY, 'M': TEXT_SECONDARY,
    }

    STRATEGY_COLORS = {
        "FOCUS_FIRE": "#ff5252", "RETREAT": "#ffd740", "FLANK": "#b388ff",
        "DEFEND": "#4da6ff", "PUSH": "#00e676", "HOLD": "#a0a0a0",
        "ROTATE": "#ff9100", "ENGAGE": "#ff5252", "DISENGAGE": "#ffd740",
    }

    GAME_TYPES = ["unknown", "auto", "fps", "fighting", "moba", "open_world"]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Game AI Control System")
        self.root.geometry("1100x780")
        self.root.configure(bg=self.BG_DARK)
        self.root.resizable(True, True)

        self._frame_data = {}
        self._lock = threading.Lock()
        self._settings_callback = None
        self._close_callback = None
        self._on_llm_correction_toggle = None

        self._build_ui()
        self._build_float_ball()
        self._setup_minimize_handler()

    def _build_ui(self):
        self._build_header()

        body = tk.Frame(self.root, bg=self.BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=self.BG_DARK)
        left.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        right = tk.Frame(body, bg=self.BG_DARK)
        right.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))

        # 左列：感知 + 决策
        self._build_perception_panel(left)
        self._build_decision_panel(left)

        # 右列：策略 + 设置
        self._build_strategy_panel(right)
        self._build_settings_panel(right)

        # 底部：键盘 + 日志
        self._build_keyboard_panel()
        self._build_log_panel()

    def _build_header(self):
        h = tk.Frame(self.root, bg=self.BG_CARD, height=48)
        h.pack(fill=tk.X, padx=8, pady=8)
        h.pack_propagate(False)

        tk.Label(h, text="GAME AI CONTROL SYSTEM", bg=self.BG_CARD,
                 fg=self.ACCENT_BLUE, font=("Consolas", 13, "bold")).pack(side=tk.LEFT, padx=12)

        self._status_dot = tk.Label(h, text="●", bg=self.BG_CARD, fg=self.ACCENT_GREEN, font=("Arial", 14))
        self._status_dot.pack(side=tk.LEFT, padx=(16, 4))
        self._status_label = tk.Label(h, text="READY", bg=self.BG_CARD, fg=self.TEXT_PRIMARY, font=("Consolas", 10))
        self._status_label.pack(side=tk.LEFT)

        self._game_type_label = tk.Label(h, text="TYPE: --", bg=self.BG_CARD,
                                          fg=self.ACCENT_PURPLE, font=("Consolas", 10))
        self._game_type_label.pack(side=tk.RIGHT, padx=12)
        self._fps_label = tk.Label(h, text="FPS: --", bg=self.BG_CARD, fg=self.ACCENT_YELLOW, font=("Consolas", 10))
        self._fps_label.pack(side=tk.RIGHT, padx=12)
        self._frame_label = tk.Label(h, text="Frame: 0", bg=self.BG_CARD, fg=self.TEXT_PRIMARY, font=("Consolas", 10))
        self._frame_label.pack(side=tk.RIGHT, padx=12)

    def _make_panel(self, parent, title, fixed_h=None):
        outer = tk.Frame(parent, bg=self.BG_DARK)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        tk.Label(outer, text=title, bg=self.BG_DARK, fg=self.ACCENT_BLUE,
                 font=("Consolas", 10, "bold")).pack(anchor=tk.W)
        pf = tk.Frame(outer, bg=self.BG_PANEL, relief=tk.RIDGE, bd=1)
        pf.pack(fill=tk.BOTH, expand=True)
        if fixed_h:
            pf.configure(height=fixed_h)
            pf.pack_propagate(False)
        return pf

    # ── 感知层 ─────────────────────────────────────────────

    def _build_perception_panel(self, parent):
        p = self._make_panel(parent, "PERCEPTION", fixed_h=240)

        tk.Label(p, text="  Enemies", bg=self.BG_PANEL, fg=self.ACCENT_RED,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 1))
        self._enemy_slots = []
        ef = tk.Frame(p, bg=self.BG_PANEL)
        ef.pack(fill=tk.X, padx=8)
        for _ in range(MAX_ENEMIES):
            lbl = tk.Label(ef, text="", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                           font=("Consolas", 9), anchor=tk.W, height=1)
            lbl.pack(fill=tk.X)
            self._enemy_slots.append(lbl)

        tk.Label(p, text="  Allies", bg=self.BG_PANEL, fg=self.ACCENT_GREEN,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 1))
        self._ally_slots = []
        af = tk.Frame(p, bg=self.BG_PANEL)
        af.pack(fill=tk.X, padx=8)
        for _ in range(MAX_ALLIES):
            lbl = tk.Label(af, text="", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                           font=("Consolas", 9), anchor=tk.W, height=1)
            lbl.pack(fill=tk.X)
            self._ally_slots.append(lbl)

        tk.Label(p, text="  Summary", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 1))
        self._summary_lbl = tk.Label(p, text="", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY,
                                     font=("Consolas", 9), anchor=tk.W, wraplength=480)
        self._summary_lbl.pack(anchor=tk.W, padx=8)

    # ── 决策层 ─────────────────────────────────────────────

    def _build_decision_panel(self, parent):
        p = self._make_panel(parent, "DECISION", fixed_h=120)

        r1 = tk.Frame(p, bg=self.BG_PANEL)
        r1.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(r1, text="Strategy:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self._strategy_lbl = tk.Label(r1, text="HOLD", bg=self.BG_PANEL, fg=self.ACCENT_BLUE,
                                      font=("Consolas", 11, "bold"))
        self._strategy_lbl.pack(side=tk.LEFT, padx=6)

        r2 = tk.Frame(p, bg=self.BG_PANEL)
        r2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r2, text="Target:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        self._target_lbl = tk.Label(r2, text="--", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY,
                                    font=("Consolas", 9))
        self._target_lbl.pack(side=tk.LEFT, padx=6)
        tk.Label(r2, text="Priority:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(16, 0))
        self._priority_lbl = tk.Label(r2, text="low", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                                      font=("Consolas", 9))
        self._priority_lbl.pack(side=tk.LEFT, padx=6)

        self._reasoning_lbl = tk.Label(p, text="", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY,
                                       font=("Consolas", 8), anchor=tk.W, wraplength=480)
        self._reasoning_lbl.pack(anchor=tk.W, padx=8, pady=(4, 0))

    # ── 设置面板（含游戏类型）──────────────────────────────

    PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1",
        "deepseek": "https://api.deepseek.com/v1", "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4", "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1", "local": "http://localhost:11434/v1", "custom": "",
    }

    def _build_settings_panel(self, parent):
        p = self._make_panel(parent, "SETTINGS", fixed_h=240)

        # API Type
        r1 = tk.Frame(p, bg=self.BG_PANEL)
        r1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r1, text="API:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._api_type_var = tk.StringVar(value="ollama")
        ttk.Combobox(r1, textvariable=self._api_type_var,
                     values=list(self.PROVIDER_URLS.keys()),
                     width=12, state="readonly").pack(side=tk.LEFT, padx=4)
        self._api_type_var.trace_add("write", lambda *a: self._on_type_changed())

        # URL
        r2 = tk.Frame(p, bg=self.BG_PANEL)
        r2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r2, text="URL:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._api_url_var = tk.StringVar(value="http://localhost:11434/v1")
        tk.Entry(r2, textvariable=self._api_url_var, bg=self.PROGRESS_BG,
                 fg=self.TEXT_PRIMARY, font=("Consolas", 9),
                 insertbackground=self.TEXT_PRIMARY, relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Key
        r3 = tk.Frame(p, bg=self.BG_PANEL)
        r3.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r3, text="Key:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._api_key_var = tk.StringVar(value="")
        tk.Entry(r3, textvariable=self._api_key_var, bg=self.PROGRESS_BG,
                 fg=self.TEXT_PRIMARY, font=("Consolas", 9), show="*",
                 insertbackground=self.TEXT_PRIMARY, relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Model + Fetch
        r4 = tk.Frame(p, bg=self.BG_PANEL)
        r4.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r4, text="Model:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value="gpt-4o-mini")
        self._model_combo = ttk.Combobox(r4, textvariable=self._model_var,
                                          values=["gpt-4o-mini"], width=20)
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._fetch_btn = tk.Button(r4, text="Fetch", bg=self.ACCENT_GREEN, fg="#000",
                                     font=("Consolas", 8, "bold"), relief=tk.FLAT,
                                     command=self._on_fetch_models, width=6)
        self._fetch_btn.pack(side=tk.RIGHT, padx=2)

        # Game Type
        r5 = tk.Frame(p, bg=self.BG_PANEL)
        r5.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r5, text="Game:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._game_type_var = tk.StringVar(value="auto")
        ttk.Combobox(r5, textvariable=self._game_type_var,
                     values=self.GAME_TYPES, width=12, state="readonly").pack(side=tk.LEFT, padx=4)

        # LLM Correction Toggle
        r6 = tk.Frame(p, bg=self.BG_PANEL)
        r6.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r6, text="LLM Fix:", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=6).pack(side=tk.LEFT)
        self._llm_correction_active = False
        self._llm_correction_btn = tk.Button(
            r6, text="OFF", bg="#555", fg="#fff",
            font=("Consolas", 9, "bold"), relief=tk.FLAT, width=8,
            command=self._toggle_llm_correction
        )
        self._llm_correction_btn.pack(side=tk.LEFT, padx=4)
        self._llm_correction_status = tk.Label(
            r6, text="", bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
            font=("Consolas", 8)
        )
        self._llm_correction_status.pack(side=tk.LEFT, padx=4)

        # Apply
        br = tk.Frame(p, bg=self.BG_PANEL)
        br.pack(fill=tk.X, padx=8, pady=(4, 6))
        self._apply_status = tk.Label(br, text="", bg=self.BG_PANEL, fg=self.ACCENT_GREEN, font=("Consolas", 9))
        self._apply_status.pack(side=tk.RIGHT, padx=4)
        tk.Button(br, text="Apply", bg=self.ACCENT_BLUE, fg="#fff",
                  font=("Consolas", 9, "bold"), relief=tk.FLAT,
                  command=self._on_apply).pack(side=tk.RIGHT)

    def _toggle_llm_correction(self):
        """切换LLM修正状态"""
        self._llm_correction_active = not self._llm_correction_active
        if self._llm_correction_active:
            self._llm_correction_btn.configure(text="ON", bg=self.ACCENT_GREEN)
            self._llm_correction_status.configure(text="60s, 5 calls", fg=self.ACCENT_GREEN)
            # 触发回调
            if self._on_llm_correction_toggle:
                self._on_llm_correction_toggle(True)
        else:
            self._llm_correction_btn.configure(text="OFF", bg="#555")
            self._llm_correction_status.configure(text="")
            if self._on_llm_correction_toggle:
                self._on_llm_correction_toggle(False)

    def set_llm_correction_status(self, call_count: int, max_calls: int):
        """更新LLM修正状态显示"""
        if self._llm_correction_active:
            self._llm_correction_status.configure(text=f"{call_count}/{max_calls}")

    def _on_type_changed(self):
        provider = self._api_type_var.get()
        url = self.PROVIDER_URLS.get(provider, "")
        self._api_url_var.set(url)

    def _on_fetch_models(self):
        self._fetch_btn.configure(text="...", state=tk.DISABLED)
        self._apply_status.configure(text="Fetching...")
        threading.Thread(target=self._fetch_models_thread, daemon=True).start()

    def _fetch_models_thread(self):
        import requests as req
        url = self._api_url_var.get().rstrip("/")
        key = self._api_key_var.get()
        api_type = self._api_type_var.get()
        models = []
        try:
            if api_type in ("ollama", "local"):
                resp = req.get(url.replace("/v1", "") + "/api/tags", timeout=5)
                if resp.ok:
                    models = [m["name"] for m in resp.json().get("models", [])]
            elif api_type == "anthropic":
                models = ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]
            else:
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                resp = req.get(url + "/models", headers=headers, timeout=10)
                if resp.ok:
                    models = sorted([m["id"] for m in resp.json().get("data", []) if m.get("id")])
        except Exception as e:
            self.root.after(0, lambda: self._apply_status.configure(text=f"Error: {e}"))
            self.root.after(0, lambda: self._fetch_btn.configure(text="Fetch", state=tk.NORMAL))
            return

        if models:
            self.root.after(0, lambda: self._model_combo.configure(values=models))
            self.root.after(0, lambda: self._model_var.set(models[0]))
            self.root.after(0, lambda: self._apply_status.configure(text=f"Found {len(models)}"))
        else:
            self.root.after(0, lambda: self._apply_status.configure(text="No models"))
        self.root.after(0, lambda: self._fetch_btn.configure(text="Fetch", state=tk.NORMAL))

    def _on_apply(self):
        s = self.get_settings()
        if self._settings_callback:
            self._settings_callback(s)
        self._apply_status.configure(text="Applied")
        self.root.after(2000, lambda: self._apply_status.configure(text=""))

    def set_settings_callback(self, cb):
        self._settings_callback = cb

    def set_llm_correction_callback(self, cb):
        """设置LLM修正切换回调"""
        self._on_llm_correction_toggle = cb

    def get_settings(self):
        return dict(api_type=self._api_type_var.get(), api_base=self._api_url_var.get(),
                    api_key=self._api_key_var.get(), model_name=self._model_var.get(),
                    game_type=self._game_type_var.get())

    def set_settings(self, s):
        for k, v in s.items():
            attr = f"_{k}_var"
            if hasattr(self, attr):
                getattr(self, attr).set(v)

    # ── 策略层 ─────────────────────────────────────────────

    def _build_strategy_panel(self, parent):
        p = self._make_panel(parent, "STRATEGY", fixed_h=120)

        tk.Label(p, text="  Mouse Velocity", bg=self.BG_PANEL, fg=self.ACCENT_BLUE,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 1))
        mf = tk.Frame(p, bg=self.BG_PANEL)
        mf.pack(fill=tk.X, padx=8)
        self._mvx_bar = self._bar(mf, "Vx", self.ACCENT_BLUE)
        self._mvy_bar = self._bar(mf, "Vy", self.ACCENT_BLUE)

        tk.Label(p, text="  Move Velocity", bg=self.BG_PANEL, fg=self.ACCENT_GREEN,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 1))
        mf2 = tk.Frame(p, bg=self.BG_PANEL)
        mf2.pack(fill=tk.X, padx=8)
        self._mvx2_bar = self._bar(mf2, "Mx", self.ACCENT_GREEN)
        self._mvy2_bar = self._bar(mf2, "My", self.ACCENT_GREEN)

    def _bar(self, parent, label, color):
        r = tk.Frame(parent, bg=self.BG_PANEL)
        r.pack(fill=tk.X, pady=1)
        tk.Label(r, text=label, bg=self.BG_PANEL, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9), width=3).pack(side=tk.LEFT)
        bg = tk.Frame(r, bg=self.PROGRESS_BG, height=14, width=200)
        bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        bg.pack_propagate(False)
        fill = tk.Frame(bg, bg=color, height=14)
        fill.place(relx=0.5, rely=0, relwidth=0, relheight=1)
        val = tk.Label(r, text="+0.000", bg=self.BG_PANEL, fg=self.TEXT_PRIMARY,
                       font=("Consolas", 9), width=7)
        val.pack(side=tk.RIGHT)
        return bg, fill, val

    def _set_bar(self, parts, v):
        bg, fill, val = parts
        a = min(abs(v), 1.0) / 2
        if v >= 0:
            fill.place(relx=0.5, rely=0, relwidth=a, relheight=1)
        else:
            fill.place(relx=0.5 - a, rely=0, relwidth=a, relheight=1)
        val.configure(text=f"{v:+.3f}")

    # ── 键盘布局面板 ───────────────────────────────────────

    def _build_keyboard_panel(self):
        """键盘布局面板（底部，跨整行）"""
        outer = tk.Frame(self.root, bg=self.BG_DARK)
        outer.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(outer, text="KEYBOARD", bg=self.BG_DARK, fg=self.ACCENT_BLUE,
                 font=("Consolas", 10, "bold")).pack(anchor=tk.W)
        p = tk.Frame(outer, bg=self.BG_PANEL, relief=tk.RIDGE, bd=1, height=240)
        p.pack(fill=tk.X)
        p.pack_propagate(False)

        kb_frame = tk.Frame(p, bg=self.BG_PANEL, padx=6, pady=4)
        kb_frame.pack(fill=tk.BOTH, expand=True)

        # 创建按键控件
        self._key_widgets = {}
        for row, col, name, colspan in KB_KEYS:
            card = tk.Frame(kb_frame, bg=self.PROGRESS_BG, relief=tk.RAISED, bd=1, height=32)
            card.grid(row=row, column=col, columnspan=colspan, padx=1, pady=1, sticky=tk.NSEW)
            card.grid_propagate(False)
            lbl = tk.Label(card, text=name, bg=self.PROGRESS_BG, fg=self.TEXT_SECONDARY,
                           font=("Consolas", 8, "bold"))
            lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self._key_widgets[name] = (card, lbl)

        # 设置行列权重
        for r in range(5):
            kb_frame.rowconfigure(r, weight=1)
        for c in range(11):
            kb_frame.columnconfigure(c, weight=1)

    # ── 日志 ───────────────────────────────────────────────

    def _build_log_panel(self):
        f = tk.Frame(self.root, bg=self.BG_CARD, height=80)
        f.pack(fill=tk.X, padx=8, pady=(0, 8))
        f.pack_propagate(False)
        tk.Label(f, text="LOG", bg=self.BG_CARD, fg=self.TEXT_SECONDARY,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self._log = tk.Text(f, bg=self.BG_DARK, fg=self.TEXT_PRIMARY, font=("Consolas", 8),
                            height=3, relief=tk.FLAT, insertbackground=self.TEXT_PRIMARY)
        self._log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        sb = tk.Scrollbar(self._log, command=self._log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.configure(yscrollcommand=sb.set)

    # ── 悬浮球 ──────────────────────────────────────────────

    def _build_float_ball(self):
        self._ball = tk.Toplevel(self.root)
        self._ball.overrideredirect(True)
        self._ball.attributes("-topmost", True)
        self._ball.attributes("-alpha", 0.92)
        sw = self._ball.winfo_screenwidth()
        self._ball.geometry(f"180x72+{sw - 200}+100")
        bg = self.BG_CARD
        self._ball.configure(bg=bg)

        frame = tk.Frame(self._ball, bg=bg, padx=8, pady=6)
        frame.pack(fill=tk.BOTH, expand=True)

        row1 = tk.Frame(frame, bg=bg)
        row1.pack(fill=tk.X)
        self._ball_dot = tk.Label(row1, text="●", bg=bg, fg=self.ACCENT_GREEN, font=("Arial", 10))
        self._ball_dot.pack(side=tk.LEFT)
        self._ball_fps = tk.Label(row1, text="FPS:--", bg=bg, fg=self.ACCENT_YELLOW, font=("Consolas", 9))
        self._ball_fps.pack(side=tk.LEFT, padx=(4, 0))
        self._ball_frame = tk.Label(row1, text="F:0", bg=bg, fg=self.TEXT_SECONDARY, font=("Consolas", 9))
        self._ball_frame.pack(side=tk.LEFT, padx=(8, 0))

        row2 = tk.Frame(frame, bg=bg)
        row2.pack(fill=tk.X, pady=(2, 0))
        self._ball_strategy = tk.Label(row2, text="HOLD", bg=bg, fg=self.ACCENT_BLUE,
                                        font=("Consolas", 10, "bold"))
        self._ball_strategy.pack(side=tk.LEFT)
        self._ball_enemy = tk.Label(row2, text="E:0", bg=bg, fg=self.ACCENT_RED, font=("Consolas", 9))
        self._ball_enemy.pack(side=tk.LEFT, padx=(10, 0))
        self._ball_ally = tk.Label(row2, text="A:0", bg=bg, fg=self.ACCENT_GREEN, font=("Consolas", 9))
        self._ball_ally.pack(side=tk.LEFT, padx=(4, 0))

        self._ball_drag_data = {"x": 0, "y": 0}
        for w in [frame, row1, row2, self._ball_dot, self._ball_fps,
                  self._ball_frame, self._ball_strategy, self._ball_enemy, self._ball_ally]:
            w.bind("<Button-1>", self._ball_start_drag)
            w.bind("<B1-Motion>", self._ball_drag)
            w.bind("<ButtonRelease-1>", self._ball_stop_drag)
            w.bind("<Double-Button-1>", lambda e: self._restore_from_ball())

        self._ball_menu = tk.Menu(self._ball, tearoff=0, bg=self.BG_PANEL, fg=self.TEXT_PRIMARY, font=("Consolas", 9))
        self._ball_menu.add_command(label="Restore", command=self._restore_from_ball)
        self._ball_menu.add_separator()
        self._ball_menu.add_command(label="Exit", command=self._on_close)
        for w in [frame, row1, row2]:
            w.bind("<Button-3>", self._show_ball_menu)

        self._ball.withdraw()
        self._is_ball_visible = False

    def _setup_minimize_handler(self):
        self.root.bind("<Unmap>", self._on_minimize)
        self.root.bind("<Map>", self._on_restore)

    def _on_minimize(self, event=None):
        if self.root.state() == "iconic":
            self._ball.deiconify()
            self._is_ball_visible = True

    def _on_restore(self, event=None):
        if self.root.state() != "iconic":
            self._ball.withdraw()
            self._is_ball_visible = False

    def _restore_from_ball(self):
        self.root.deiconify()
        self.root.lift()
        self._ball.withdraw()
        self._is_ball_visible = False

    def _ball_start_drag(self, event):
        self._ball_drag_data["x"] = event.x
        self._ball_drag_data["y"] = event.y

    def _ball_drag(self, event):
        x = self._ball.winfo_x() + event.x - self._ball_drag_data["x"]
        y = self._ball.winfo_y() + event.y - self._ball_drag_data["y"]
        self._ball.geometry(f"+{x}+{y}")

    def _ball_stop_drag(self, event):
        self._ball_drag_data = {"x": 0, "y": 0}

    def _show_ball_menu(self, event):
        self._ball_menu.post(event.x_root, event.y_root)

    def _update_float_ball(self, d):
        if not self._is_ball_visible:
            return
        self._ball_fps.configure(text=f"FPS:{d.get('fps', 0):.0f}")
        self._ball_frame.configure(text=f"F:{d.get('frame_id', 0)}")
        strat = d.get('strategy', 'HOLD')
        self._ball_strategy.configure(text=strat, fg=self.STRATEGY_COLORS.get(strat, self.TEXT_PRIMARY))
        self._ball_enemy.configure(text=f"E:{len(d.get('enemies', []))}")
        self._ball_ally.configure(text=f"A:{len(d.get('allies', []))}")

    # ── 数据刷新 ───────────────────────────────────────────

    def update_frame(self, data):
        with self._lock:
            self._frame_data = data.copy()

    def _refresh_ui(self):
        with self._lock:
            d = self._frame_data
        if not d:
            self.root.after(50, self._refresh_ui)
            return

        # Header
        self._frame_label.configure(text=f"Frame: {d.get('frame_id', 0)}")
        self._fps_label.configure(text=f"FPS: {d.get('fps', 0):.1f}")
        self._status_label.configure(text="RUNNING")
        self._status_dot.configure(fg=self.ACCENT_GREEN)
        self._game_type_label.configure(text=f"TYPE: {d.get('game_type', '--')}")

        # Enemies
        enemies = d.get('enemies', [])
        for i in range(MAX_ENEMIES):
            lbl = self._enemy_slots[i]
            if i < len(enemies):
                e = enemies[i]
                hp = e.get('hp', 0)
                c = self.ACCENT_RED if hp < 0.3 else self.ACCENT_YELLOW if hp < 0.7 else self.ACCENT_GREEN
                lbl.configure(text=f"  {e.get('id', '?'):4s} HP:{hp*100:3.0f}%  {e.get('hp_status', ''):8s}  {e.get('pos', '')}",
                              fg=c, bg=self.BG_CARD)
            else:
                lbl.configure(text="", fg=self.TEXT_SECONDARY, bg=self.BG_PANEL)

        # Allies
        allies = d.get('allies', [])
        for i in range(MAX_ALLIES):
            lbl = self._ally_slots[i]
            if i < len(allies):
                a = allies[i]
                hp = a.get('hp', 0)
                c = self.ACCENT_RED if hp < 0.3 else self.ACCENT_YELLOW if hp < 0.7 else self.ACCENT_GREEN
                lbl.configure(text=f"  {a.get('id', '?'):4s} HP:{hp*100:3.0f}%  {a.get('hp_status', ''):8s}  {a.get('pos', '')}",
                              fg=c, bg=self.BG_CARD)
            else:
                lbl.configure(text="", fg=self.TEXT_SECONDARY, bg=self.BG_PANEL)

        self._summary_lbl.configure(text=d.get('summary', ''))

        # Decision
        strat = d.get('strategy', 'HOLD')
        self._strategy_lbl.configure(text=strat, fg=self.STRATEGY_COLORS.get(strat, self.TEXT_PRIMARY))
        self._target_lbl.configure(text=d.get('target', '--'))
        pri = d.get('priority', 'low')
        pc = self.ACCENT_RED if pri == 'high' else self.ACCENT_YELLOW if pri == 'medium' else self.TEXT_SECONDARY
        self._priority_lbl.configure(text=pri, fg=pc)
        self._reasoning_lbl.configure(text=d.get('reasoning', ''))

        # Strategy bars
        mx, my = d.get('mouse_velocity', (0, 0))
        self._set_bar(self._mvx_bar, mx)
        self._set_bar(self._mvy_bar, my)
        vx, vy = d.get('move_velocity', (0, 0))
        self._set_bar(self._mvx2_bar, vx)
        self._set_bar(self._mvy2_bar, vy)

        # Keyboard - 40维按键更新
        btns = d.get('buttons', [False] * 36)
        for i, name in enumerate(KEY_NAMES):
            if name in self._key_widgets:
                card, lbl = self._key_widgets[name]
                on = i < len(btns) and btns[i]
                c = self.KEY_COLORS.get(name, self.ACCENT_BLUE)
                if on:
                    card.configure(bg=c)
                    lbl.configure(text=name, bg=c, fg="#fff")
                else:
                    card.configure(bg=self.PROGRESS_BG)
                    lbl.configure(text=name, bg=self.PROGRESS_BG, fg=self.TEXT_SECONDARY)

        # Log
        msg = d.get('log_message', '')
        if msg:
            self._log.insert(tk.END, msg + "\n")
            self._log.see(tk.END)
            lines = int(self._log.index('end-1c').split('.')[0])
            if lines > 60:
                self._log.delete('1.0', f'{lines - 40}.0')

        self._update_float_ball(d)
        self.root.after(50, self._refresh_ui)

    def _on_close(self):
        if self._close_callback:
            self._close_callback()
        self._ball.destroy()
        self.root.destroy()

    def start(self):
        self._refresh_ui()
        self.root.mainloop()

    def stop(self):
        self._ball.destroy()
        self.root.quit()

    def set_close_callback(self, cb):
        self._close_callback = cb
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
