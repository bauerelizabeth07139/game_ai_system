"""全面功能测试 - 141维状态向量 + 40维输出"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

passed = 0
failed = 0

def test(name):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")

def ok(msg):
    global passed
    passed += 1
    print(f"  [PASS] {msg}")

def fail(msg, e=None):
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")
    if e:
        print(f"         {e}")

# ============================================================
test("Perception - ColorEngine")
try:
    from src.perception.color_engine import ColorEngine
    engine = ColorEngine()
    img = np.zeros((540, 960, 3), dtype=np.uint8)
    img[100:120, 200:250] = (0, 0, 200)
    img[150:165, 600:640] = (0, 0, 180)
    img[200:215, 400:440] = (0, 200, 0)
    result = engine.process(img)
    ok(f"enemies:{len(result['enemies'])}, allies:{len(result['allies'])}")
except Exception as e:
    fail("ColorEngine", e)

# ============================================================
test("Perception - SemanticLabeler")
try:
    from src.perception.semantic_labeler import SemanticLabeler
    labeler = SemanticLabeler()
    semantic = labeler.generate_frame_data(1, result['enemies'], result['allies'], result['image_size'])
    ok(f"summary: {semantic['summary'][:40]}")
except Exception as e:
    fail("SemanticLabeler", e)

# ============================================================
test("Perception - TimeBuffer (3 timepoints [0,-2,-6])")
try:
    from src.perception.time_buffer import TimeBuffer
    buf = TimeBuffer({'buffer_size': 50, 'mlp_history_frames': [0, -2, -6], 'decision_history_frames': [0, -8, -42]})
    for i in range(10):
        sv = np.random.randn(141).astype(np.float32)
        buf.push({'frame_id': i, 'state_vector': sv, 'semantic_data': {'frame_id': i}})
    mlp_data = buf.sample_for_mlp()
    assert mlp_data.shape == (423,), f"Expected (423,), got {mlp_data.shape}"
    dec_data = buf.sample_for_decision()
    assert len(dec_data) == 3, f"Expected 3 decision frames, got {len(dec_data)}"
    ok(f"MLP:{mlp_data.shape}, Decision:{len(dec_data)}frames")
except Exception as e:
    fail("TimeBuffer", e)

# ============================================================
test("Perception - OCREngine (disabled)")
try:
    from src.perception.ocr_engine import OCREngine
    ocr = OCREngine({'enabled': False})
    texts = ocr.extract_text(img)
    assert texts == []
    ok("Disabled mode returns []")
except Exception as e:
    fail("OCREngine", e)

# ============================================================
test("Decision - LLMClient")
try:
    from src.decision.llm_client import LLMClient
    llm = LLMClient({'interval_frames': 50, 'api_base': 'http://localhost:11434/v1', 'game_type': 'fps'})
    sim1 = llm.simulate_decision({'enemies': [{'id': 'E1', 'hp': 0.2}], 'allies': [{'id': 'A1', 'hp': 0.8}]})
    sim2 = llm.simulate_decision({'enemies': [{'id': 'E1', 'hp': 0.8}], 'allies': [{'id': 'A1', 'hp': 0.1}, {'id': 'A2', 'hp': 0.15}]})
    assert sim1['strategy'] == 'FOCUS_FIRE'
    assert sim2['strategy'] == 'RETREAT'
    ok(f"FOCUS_FIRE/RETREAT correct, game_type={llm.game_type}")
except Exception as e:
    fail("LLMClient", e)

# ============================================================
test("Decision - Game Type Prompts")
try:
    from src.decision.llm_client import GAME_TYPE_PROMPTS
    assert 'fps' in GAME_TYPE_PROMPTS
    assert 'fighting' in GAME_TYPE_PROMPTS
    assert 'moba' in GAME_TYPE_PROMPTS
    assert 'open_world' in GAME_TYPE_PROMPTS
    assert 'auto' in GAME_TYPE_PROMPTS
    ok(f"{len(GAME_TYPE_PROMPTS)} game type prompts loaded")
except Exception as e:
    fail("Game Type Prompts", e)

# ============================================================
test("Decision - PromptBuilder")
try:
    from src.decision.prompt_builder import PromptBuilder
    pb = PromptBuilder()
    msg = pb.build_user_message(100, semantic['enemies'], semantic['allies'], semantic['summary'], "test")
    assert "F:100" in msg
    ok(f"Format correct, {len(msg)} chars")
except Exception as e:
    fail("PromptBuilder", e)

# ============================================================
test("Strategy - MLPModel (456 input, 40 output)")
try:
    import torch
    from src.strategy.mlp_model import MLPModel, STRATEGY_TO_IDX
    model = MLPModel()
    assert model.INPUT_DIM == 456, f"Expected INPUT_DIM=456, got {model.INPUT_DIM}"
    assert model.OUTPUT_DIM == 40
    state = torch.randn(1, 423)  # 141*3
    idx = torch.LongTensor([STRATEGY_TO_IDX['FOCUS_FIRE']])
    gt = torch.FloatTensor([[0.0]])  # fps
    with torch.no_grad():
        out = model(state, idx, gt)
    assert out.shape == (1, 40), f"Expected (1,40), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "All Sigmoid [0,1]"
    ok(f"456->512->1024->512->40, all Sigmoid[0,1]")
except Exception as e:
    fail("MLPModel", e)

# ============================================================
test("Strategy - Inference (40-dim, 141 state)")
try:
    from src.strategy.inference import StrategyInference
    si = StrategyInference()
    si.set_strategy('FOCUS_FIRE')
    si.set_game_type('fps')
    action = si.infer(np.random.randn(423).astype(np.float32))
    assert action.shape == (40,), f"Expected (40,), got {action.shape}"
    assert (action >= 0).all() and (action <= 1).all()
    ok(f"40-dim output, game_type={si.current_game_type}")
except Exception as e:
    fail("StrategyInference", e)

# ============================================================
test("Execution - PIDController")
try:
    from src.execution.pid_controller import PIDController
    pid = PIDController({'Kp': 1.2, 'Ki': 0.01, 'Kd': 0.1})
    for _ in range(5):
        v = pid.update(np.array([0.5, 0.5]))
    err = abs(v[0] - 0.5)
    ok(f"Velocity converged, error={err:.4f}")
except Exception as e:
    fail("PIDController", e)

# ============================================================
test("Execution - NoiseInjector")
try:
    from src.execution.noise_injector import NoiseInjector
    ni = NoiseInjector({'frequency': 8.0, 'amplitude': 0.02})
    noises = [ni.get_noise() for _ in range(10)]
    ok(f"Range: [{min(noises):+.4f}, {max(noises):+.4f}]")
except Exception as e:
    fail("NoiseInjector", e)

# ============================================================
test("Execution - HAL (40-dim)")
try:
    from src.execution.hal import HAL
    hal = HAL({'mouse_sensitivity': 100}, real_input=False)
    hal.start()
    action = np.zeros(40, dtype=np.float32)
    action[0] = 0.7
    action[1] = 0.3
    action[2] = 0.6
    action[3] = 0.8
    action[4] = 1.0  # LMB
    action[7] = 1.0  # W
    action[14] = 1.0  # E
    action[26] = 1.0  # Q
    hal.execute(action)
    time.sleep(0.3)
    hal.stop()
    summary = hal.format_action_summary(action)
    assert "LMB" in summary and "W" in summary
    ok(f"40-dim, {summary}")
except Exception as e:
    fail("HAL", e)

# ============================================================
test("Full Pipeline (10 frames, 141 state)")
try:
    color = ColorEngine()
    label = SemanticLabeler()
    buffer = TimeBuffer({'buffer_size': 50, 'mlp_history_frames': [0, -2, -6], 'decision_history_frames': [0, -8, -42]})
    strat = StrategyInference()
    strat.set_game_type('fps')
    h = HAL({'mouse_sensitivity': 100}, real_input=False)
    h.start()
    strat.set_strategy('DEFEND')

    for frame in range(10):
        test_img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
        test_img[100:120, 200:250] = (0, 0, 200)
        det = color.process(test_img)
        sem = label.generate_frame_data(frame, det['enemies'], det['allies'], det['image_size'])
        sv = np.zeros(141, dtype=np.float32)
        buffer.push({'frame_id': frame, 'state_vector': sv, 'semantic_data': sem})
        mlp_in = buffer.sample_for_mlp()
        assert mlp_in.shape == (423,), f"Buffer shape: {mlp_in.shape}"
        action = strat.infer(mlp_in)
        assert action.shape == (40,), f"Action shape: {action.shape}"
        h.execute(action)

    time.sleep(0.5)
    h.stop()
    ok(f"10 frames, buffer={len(buffer)}, game_type={strat.current_game_type}")
except Exception as e:
    fail("Full Pipeline", e)

# ============================================================
print(f"\n{'='*60}")
print(f"Result: {passed} PASS, {failed} FAIL")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
